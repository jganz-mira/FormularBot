from typing import Optional, Tuple, List, Dict, Any, Literal
import os
import json
from .translator import translate_from_de
from gradio import ChatMessage
from difflib import SequenceMatcher
import difflib
import unicodedata
import re
from pydantic import BaseModel
from .llm_validator_service import LLMValidatorService
from openai import OpenAI
import cv2 
import pytesseract
from src.validator_helper import response_to_dict


def load_forms(form_path:str, validator_map:Dict[str,callable]):
    forms = {}
    for fname in os.listdir(form_path):
        if fname.endswith(".json"):
            with open(os.path.join(form_path, fname), encoding="utf-8") as f:
                form_conf = json.load(f)
            validator_class = validator_map[form_conf["validators"]]
            form_conf["validators"] = validator_class
            form_key = fname.rsplit(".", 1)[0]
            forms[form_key] = form_conf
    return forms

# def next_slot_index(
#     slots_def: List[Dict[str, Any]],
#     state: Dict[str, Any]
# ) -> Tuple[Optional[int], Dict[str, Any]]:
#     """
#     Determines the index of the next slot to ask based on the slot definitions
#     and current state, skipping slots whose 'condition' is not fulfilled.

#     If a slot has a 'condition', it will only be considered if the condition is met.
#     If the condition is not fulfilled, the slot will be skipped and assigned an
#     empty string as its value in the state to prevent assertion errors for dependent slots.

#     Args:
#         slots_def: A list of slot definition dictionaries. Each slot may include a 
#                    'condition' that determines whether it should be shown.
#         state: A dictionary containing:
#             - "idx" (int): the current index to start checking from,
#             - "responses" (Dict[str, Dict[str, str]]): previous answers,
#               where each key is a slot name, and the value is a dict with:
#                 - "value" (str): the user's input,
#                 - "target_filed_name" (str): the field name in the output.

#     Returns:
#         A tuple containing:
#             - The index (int) of the next askable slot, or None if no such slot exists.
#             - The updated state dictionary.
#     """

#     # current index to start searching from
#     start_idx = state["idx"]
#     responses = state["responses"]


#     for i in range(start_idx, len(slots_def)):
#         slot_def = slots_def[i]
#         cond = slot_def.get("condition")
#         # if a condition is defined, skip unless it's fulfilled
#         if cond:
#             assert responses.get(cond["slot_name"]) is not None, "The slot on which a slot is conditioned needs to be asked first!"
#             # here one can define other "conditions" to check for
#             if cond["slot_value"] == 'not empty':
#                 prev_val = responses.get(cond["slot_name"])
#                 if prev_val['value'] == "":
#                     # If conditional slot is skipped, write an empty string so if another slot depends on that one, no assertion is triggered
#                     state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
#                     # Afterward, we move on to the next slot
#                     continue

#             # if the slot_value contains a list, then the conditional slot will only be active if the the
#             # slot_value of the other filed is in this list
#             elif isinstance(cond['slot_value'], list):
#                 prev_val = responses.get(cond["slot_name"])
#                 if prev_val['value'] not in cond["slot_value"]:
#                     state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
#                     continue
#             # fallback condition, check for equality to "slot_value" defined in json
#             else:
#                 prev_val = responses.get(cond["slot_name"])
#                 if prev_val['value'] != cond["slot_value"]:
#                     state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
#                     continue
#         # return next slot index and the updated state
#         return i, state
#     # if there is no next slot (end of document), next slot index i is none
#     return None, state

def next_slot_index(
    slots_def: List[Dict[str, Any]],
    state: Dict[str, Any]
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Liefert den Index des nächsten abzufragenden Slots.
    Regeln:
      - Bereits beantwortete Slots (value != "" und nicht None) werden übersprungen.
      - Slots, deren 'condition' NICHT erfüllt ist, werden 'soft-geskippt':
        state['responses'][slot] = {'value': '', 'target_filed_name': ..., 'locked': True}
        und übersprungen.
      - Leere, aber NICHT gelockte Antworten gelten als "noch offen" → werden gefragt.
    """
    start_idx = int(state.get("idx", 0) or 0)
    responses = state.setdefault("responses", {})

    def _answered(slot_name: str) -> bool:
        """True, wenn Slot bereits sinnvoll gefüllt ODER durch Condition-Lock gesperrt ist."""
        entry = responses.get(slot_name)
        if not entry:
            return False
        if entry.get("locked"):
            return True
        val = entry.get("value")
        # leere Strings/Listen/Dicts = nicht beantwortet
        return val not in (None, "", [], {})

    for i in range(start_idx, len(slots_def)):
        slot_def = slots_def[i]
        slot_name = slot_def["slot_name"]
        cond = slot_def.get("condition")

        # Bereits beantwortet? -> weiter
        if _answered(slot_name):
            continue

        # Condition prüfen
        if cond:
            dep = cond["slot_name"]
            assert responses.get(dep) is not None, \
                "The slot on which a slot is conditioned needs to be asked first!"

            dep_val = responses.get(dep, {}).get("value")
            should_ask = False

            if cond["slot_value"] == "not empty":
                should_ask = bool(dep_val)
            elif isinstance(cond["slot_value"], list):
                should_ask = dep_val in cond["slot_value"]
            else:
                should_ask = (dep_val == cond["slot_value"])

            if not should_ask:
                # soft-skip: als gelockt markieren, damit wir später nicht wiederkommen
                responses[slot_name] = {
                    "value": "",
                    "target_filed_name": slot_def.get("filed_name"),
                    "locked": True
                }
                continue  # nächsten Slot prüfen

        # Wenn wir hier sind: Slot ist dran
        return i, state

    # Keine weiteren Slots
    return None, state


def print_summary(state: Dict[str, Any], forms: Dict[str, Any]) -> None:
    """
    Druckt alle gesammelten Antworten am Ende des Dialogs aus.
    Funktioniert sowohl mit Slots als strings als auch mit dict-Definitionen.

    Args:
        state: Der Chat-State dict mit
            - "form_type": key des ausgefüllten Formulars in `forms`
            - "lang": Sprachcode (z.B. "de")
            - "responses": dict slot_name → Antwort
        forms: Dict mapping form_type → Form-Konfiguration (mit "slots" und "prompt_map")
    """
    form_type = state.get("form_type")
    if not form_type:
        print("Kein ausgefülltes Formular gefunden.")
        return

    form_conf = forms[form_type]
    lang      = state.get("lang", "de")
    responses = state.get("responses", {})

    print(f"\n--- Zusammenfassung für Formular '{form_type}' ---")
    for sd in form_conf["slots"]:
        # Wenn sd ein dict ist, slot_name extrahieren, sonst sd selbst verwenden
        if isinstance(sd, dict):
            slot_name = sd["slot_name"]
        else:
            slot_name = sd

        # Label aus prompt_map holen (fallback auf slot_name)
        # label = form_conf["prompt_map"][lang].get(slot_name, slot_name)
        label = sd.get('prompt')
        # Antwort aus responses (falls nicht vorhanden, Hinweis ausgeben)
        answer = responses.get(slot_name, "(nicht ausgefüllt)")

        print(f"{label} {answer}")
    print("--- Ende der Zusammenfassung ---\n")

def map_yes_no_to_bool(selection: str) -> str:
    """
    Map a German yes/no choice to string "true"/"false".
    """
    norm = selection.strip().lower()
    if norm == "ja":
        return "true"
    if norm == "nein":
        return "false"
    return selection  # fallback: unverändert

def save_responses_to_json(state: dict, output_path: str):
    """
    Liest alle Antworten aus state['responses'] aus und schreibt sie
    als JSON im Format { slot_name: { value, target_filed_name } }.
    """
    responses = state.get("responses", {})
    out_data = {}

    for slot_name, details in responses.items():
        # Hole value und target_filed_name, falls vorhanden
        value = details.get("value")
        target = details.get("target_filed_name")
        choices = details.get("choices",None)
        check_box_condition = details.get("check_box_condition",None)
        # Nur Slot eintragen, wenn mindestens value vorhanden ist
        if value is not None:
            out_data[slot_name] = {
                "value": value,
                "target_filed_name": target,
                "choices": choices,
                "check_box_condition": check_box_condition
            }

    result = {
        "form_type": state.get("form_type"),
        "lang":       state.get("lang"),
        "data":       out_data,
        "pdf_file": state.get("pdf_file")
    }

    # Verzeichnis anlegen, falls nicht existent
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Schreiben
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Antworten gespeichert in {output_path}")

def utter_message_with_translation(history, prompt:str, target_lang:str, source_lang:str = None):
    if target_lang == 'de':
        history.append(ChatMessage(role="assistant", content = prompt))
    elif source_lang != None and target_lang == source_lang:
        history.append(ChatMessage(role="assistant", content = prompt))
    elif target_lang == None: # no message selected yet -> wizards utters message in user language
        history.append(ChatMessage(role="assistant", content = prompt))
    else:
        history.append(ChatMessage(role='assistant', content = translate_from_de(prompt,target_lang)))
    return history

def compose_prompt_for_slot(slot_def: Dict[str, Any]) -> str:
    """
    Baut den Prompt für einen Slot:
    - nimmt 'prompt' (oder 'description' als Fallback)
    - hängt bei choice-Slots die nummerierten Optionen an
    - hängt (falls vorhanden) 'additional_information' an (Markdown/HTML möglich)
    """
    prompt = slot_def.get("prompt", slot_def.get("description", "")) or ""

    # Choices anhängen
    # if slot_def.get("slot_type") == "choice":
    #     opts = slot_def.get("choices", [])
    #     if opts:
    #         prompt += "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))

    # # Zusätzliche Infos anhängen
    # add_info = slot_def.get("additional_information")
    # if add_info:
    #     if isinstance(add_info, list):
    #         extra = "\n\n" + "\n".join(add_info)
    #     else:
    #         extra = "\n\n" + str(add_info)
    #     prompt += extra

    return prompt


_UMLAUT_MAP = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
})

def _strip_accents(s: str) -> str:
    # Entfernt Diakritika (z. B. französische Akzente)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _normalize(s: str) -> str:
    s = s.strip().translate(_UMLAUT_MAP)
    s = _strip_accents(s)
    s = s.lower()
    # Nur Buchstaben/Ziffern/Leerzeichen behalten
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Mehrfachspaces vereinheitlichen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()  # 0..1

# --- Fuzzy Choice Matching ---

def _best_choice_match(user_text: str, choices: List[str]) -> Tuple[Optional[str], float]:
    """
    Liefert (beste_choice, score). Score 1.0 = perfekt, 0.0 = kein Match.
    Heuristik:
      1) exakter Normalized-Match
      2) Token-Subset/Substring-Match
      3) Fuzzy-Score (difflib)
    """
    ut = _normalize(user_text)
    if not ut:
        return None, 0.0
    ut_tokens = set(ut.split())

    best: Tuple[Optional[str], float] = (None, 0.0)

    for choice in choices:
        ct = _normalize(choice)
        if not ct:
            continue
        # 1) Exakt?
        if ut == ct:
            return choice, 1.0

        # 2) Token-Subset / Substring
        ct_tokens = set(ct.split())
        # Nutzer gibt z.B. "uebernahme" und Choice ist "uebernahme erbfolge kauf pacht"
        if ut in ct or ut_tokens.issubset(ct_tokens):
            score = 0.92  # hoch, aber nicht perfekt
            if score > best[1]:
                best = (choice, score)

        # 3) Fuzzy-Ähnlichkeit (Tippfehler)
        # Vergleiche sowohl Gesamtausdruck als auch gegen das längste Wort in der Choice
        sim_full = _similar(ut, ct)
        sim_word = max((_similar(ut, w) for w in ct_tokens), default=0.0)
        score = max(sim_full, sim_word)
        if score > best[1]:
            best = (choice, score)

    return best

# --- Öffentliche API ---

def ensure_string(value) -> str:
    if not isinstance(value, str):
        return str(value)
    return value


def valid_choice_slot(message: str, slot_def: Dict[str, Any], cutoff: float = 0.85) -> bool:
    """
    Wie deine Originalfunktion, aber mit fuzzy Matching.
    - Ziffern-Index (1-basiert) bleibt erhalten.
    - Textvergleich nutzt Normalisierung, Token-Containment und difflib-Score.
    """
    choices: List[str] = slot_def.get("choices", [])
    text = message.strip()

    # 1) Digit input als Index?
    if text.rstrip(".").isdigit():
        idx = int(text.rstrip(".")) - 1
        if 0 <= idx < len(choices):
            print(f"Index match: '{text}' -> '{choices[idx]}'")
            return message , True

    # 2) Fuzzy-Text-Match
    match, score = _best_choice_match(text, choices)
    if score < cutoff:
        print(f"Fuzzy match failed: '{text}' -> '{match}' (score {score:.2f})")
        # llm fallback
        match, score = llm_based_match(message, choices)
        print(f"LLM match: '{text}' -> '{match}' (score {score:.2f})")
        return match, score >= 0.5
    
    return match, score >= cutoff

def fuzzy_choice_match(message: str, choices: List[str], cutoff: float = 0.85) -> Optional[str]:
    """
    Findet den am besten passenden Choice-Eintrag mit fuzzy matching.
    - cutoff: Score-Schwelle zwischen 0 und 1 (Standard 0.85)
    - Gibt den gematchten Choice zurück oder None
    """
    text = message.strip()
    match, score = _best_choice_match(text, choices)
    print(f"Fuzzy match: '{text}' -> '{match}' (score {score:.2f})")
    if score >= cutoff:
        return match
    else: # llm fallback
        match, llm_score = llm_based_match(message, choices)
        print(f"LLM match: '{text}' -> '{match}' (score {llm_score:.2f})")
        if llm_score >= 0.5:
            return match

class ActivityCheckResponse(BaseModel):
    match: int
    score: float

def llm_based_match(message:str, choices: List[str]) -> Dict:
    '''performs choice matching based on an llm call'''

    choice_text = "\n".join(f"{i+1}. {o}" for i, o in enumerate(choices))

    system_prompt = (
        "Du bist ein präziser Intent und Choice-Klassifikator.\n"
        f"Das ist die Liste der möglichen Optionen:{choice_text}.\n"
        "Erkenne anhand der Nutzereingabe, welche Option der Nutzer gemeint hat.\n"
        "Gib die passende option exakt zurück (match) und einen Score (score) zwischen 0.0 und 1.0, wobei alles unter 0.5 kein match ist, ab 0.5 eher sicher, ab 0.75 ziemlich sicher und 1.0 absolut sicher ist.\n"
        "**Halte dich exakt an die vorgegebenen Optionen und erfinde keine neuen**.\n"
    )

    llm_service = LLMValidatorService()
    response = llm_service.validate_openai_structured_output(
        system_prompt=system_prompt,
        user_input=message,
        json_schema=ActivityCheckResponse,
        model = 'gpt-4o-mini',
        client = OpenAI()
    )

    score = response.output_parsed.score
    match = response.output_parsed.match
    return match, score

class Name(BaseModel):
    family_name: str
    given_name: str
    city: str
    birth_date: str

class HRA(BaseModel):
    authority: str
    hra_number: str
    company_name: str
    legal_type:str
    address:str
    activity:str
    ceo:List[Name]


def extract_information_HRA_info_from_img(img)->Dict:
    if isinstance(img,list):
        extracted_text = ''
        for i in img:
            text = pytesseract.image_to_string(i, lang='deu')
            
            extracted_text += text
    else:
        extracted_text = pytesseract.image_to_string(img,lang='deu')
    # post processing with llm
    system_prompt = (f"Du bist ein hochpräzises Textexraktionsmodell welches aus einem OCR string eines Bildes Informationen extrahiert. Extrahiere aus dem folgenden Str:\n"
                 "Den Namen des Registergerichts / Handelsregister (authority), die Handelsregisternummer (HRA), den Namen der Firma (company_name), die Geschäftsform (legal_type) (GmbH, GDR, etc.), die Adresse des Sitzes/Niederlassung/Geschäftsanschrift (address), den Gegenstand des Unternehmens (activity), den Nachnamen(family_name), Vornamen(given_name) (im Text findest du immer Nachname, Vorname, Wohnort, Geburtsdatum),  Wohnort (city) und das Geburtsdatum (birthdate)(Nur das Datum im Format: TT.MM.JJJJ) des Geschäftsführers (CEO) (lege diese angeben in einer json ab.).\n"
                 "Halte dich strikt an das JSON Format, erfinde unter keinen Umständen Angaben. Falls eine Angabe fehlt, lass das entsprechende Feld leer.")

    llm_service = LLMValidatorService()

    response = llm_service.validate_openai_structured_output(
        system_prompt=system_prompt,
        user_input=extracted_text,
        model="gpt-4.1-mini",
        client=OpenAI(),
        json_schema = HRA
    )

    return response_to_dict(response)

class Address(BaseModel):
    postalcode: str
    city: str
    street_name: str
    street_number: str

class IDCard(BaseModel):
    given_name: str
    family_name: str
    birth_date: str
    birth_place: str
    germany:bool
    nationality:str
    address:Address

def extract_information_id_card(img)->Dict:
    if isinstance(img,list):
        extracted_text = ''
        for i in img:
            text = pytesseract.image_to_string(i, lang='deu')
            
            extracted_text += text
    else:
        extracted_text = pytesseract.image_to_string(img,lang='deu')
        
    # extracted_text = pytesseract.image_to_string(img, lang='deu')
    # post processing with llm
    system_prompt = (f"Du bist ein hochpräzises Textexraktionsmodell welches aus einem OCR string eines Bildes eines Personalausweises Informationen extrahiert. Extrahiere aus dem folgenden Str:\n"
                 "Das Staatsangehörigkeit (nationality), den Geburtsort (birth_place), den Vornamen (surname), Nachnamen (given_name), Geburtsdatum (birth_date), und die Adresse (address). Extrahiere für die Adresse die Postleitzahl (postalcode), den Ortsnamen (city), den Straßennamen (street_name) und die Hausnummer (street_number). Wenn die Staatsangehörigkeit **DEUTSCH** ist, dann setze germany auf true, sonst false.\n"
                 "Halte dich strikt an das JSON Format, erfinde unter keinen Umständen Angaben. Falls eine Angabe fehlt, lass das entsprechende Feld leer.")

    llm_service = LLMValidatorService()

    response = llm_service.validate_openai_structured_output(
        system_prompt=system_prompt,
        user_input=extracted_text,
        model="gpt-4.1-mini",
        client=OpenAI(),
        json_schema = IDCard
    )

    return response_to_dict(response)
