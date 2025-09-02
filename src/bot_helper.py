from typing import Optional, Tuple, List, Dict, Any
import os
import json


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

def next_slot_index(
    slots_def: List[Dict[str, Any]],
    state: Dict[str, Any]
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Determines the index of the next slot to ask based on the slot definitions
    and current state, skipping slots whose 'condition' is not fulfilled.

    If a slot has a 'condition', it will only be considered if the condition is met.
    If the condition is not fulfilled, the slot will be skipped and assigned an
    empty string as its value in the state to prevent assertion errors for dependent slots.

    Args:
        slots_def: A list of slot definition dictionaries. Each slot may include a 
                   'condition' that determines whether it should be shown.
        state: A dictionary containing:
            - "idx" (int): the current index to start checking from,
            - "responses" (Dict[str, Dict[str, str]]): previous answers,
              where each key is a slot name, and the value is a dict with:
                - "value" (str): the user's input,
                - "target_filed_name" (str): the field name in the output.

    Returns:
        A tuple containing:
            - The index (int) of the next askable slot, or None if no such slot exists.
            - The updated state dictionary.
    """

    # current index to start searching from
    start_idx = state["idx"]
    responses = state["responses"]


    for i in range(start_idx, len(slots_def)):
        slot_def = slots_def[i]
        cond = slot_def.get("condition")
        # if a condition is defined, skip unless it's fulfilled
        if cond:
            assert responses.get(cond["slot_name"]) is not None, "The slot on which a slot is conditioned needs to be asked first!"
            # here one can define other "conditions" to check for
            if cond["slot_value"] == 'not empty':
                prev_val = responses.get(cond["slot_name"])
                if prev_val['value'] == "":
                    # If conditional slot is skipped, write an empty string so if another slot depends on that one, no assertion is triggered
                    state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
                    # Afterward, we move on to the next slot
                    continue

            # if the slot_value contains a list, then the conditional slot will only be active if the the
            # slot_value of the other filed is in this list
            elif isinstance(cond['slot_value'], list):
                prev_val = responses.get(cond["slot_name"])
                if prev_val['value'] not in cond["slot_value"]:
                    state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
                    continue
            # fallback condition, check for equality to "slot_value" defined in json
            else:
                prev_val = responses.get(cond["slot_name"])
                if prev_val['value'] != cond["slot_value"]:
                    state["responses"][slot_def["slot_name"]] = {"value": "", "target_filed_name": slot_def["filed_name"]}
                    continue
        # return next slot index and the updated state
        return i, state
    # if there is no next slot (end of document), next slot index i is none
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
