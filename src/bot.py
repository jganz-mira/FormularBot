
from typing import Optional, Tuple, List, Dict, Any
import os
from .validators import BaseValidators, GewerbeanmeldungValidators
from .bot_helper import load_forms, next_slot_index, print_summary, map_yes_no_to_bool, save_responses_to_json, utter_message_with_translation, compose_prompt_for_slot, valid_choice_slot, fuzzy_choice_match
from .llm_validator_service import LLMValidatorService
from openai import OpenAI
from gradio import ChatMessage
from .pdf_backend import GenericPdfFiller
from .wizards import LanguageWizard, LanguageWizardState, FormSelectionWizard, FormSelectionWizardState, ActivityWizard, ActivityWizardState 
from .translator import translate_from_de, translate_to_de, EDIT_CMDS, instruction_msgs

# form_path = "../forms/ge"   # Passe ggf. den Pfad an
# Basisverzeichnis bestimmen
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forms_path = os.path.join(base_path, 'forms', 'ge')

# OpenAI API Key aus Datei laden
key_path = os.path.join(base_path, "../.key")
try:
    with open(key_path, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()
        os.environ.setdefault("OPENAI_API_KEY", api_key)
except FileNotFoundError:
    raise RuntimeError(f"OpenAI key file not found at {key_path}")

validator_map = {
    "BaseValidators": BaseValidators,
    "GewerbeanmeldungValidators": GewerbeanmeldungValidators(),
}

FORMS = load_forms(
    form_path = forms_path,
    validator_map = validator_map
)

# EDIT_CMDS = {"ändern", "korrigieren", "korrektur", "update"} # diese kommen später an einen anderen Ort


def build_wizard_from_state(name: str, data: dict):
    if name == "language_wizard":
        return LanguageWizard(LanguageWizardState(**(data or {})))
    if name == "form_selection_wizard":
        return FormSelectionWizard(FormSelectionWizardState(**(data or {})))
    # if name == "activity_wizard":  # NEU
    #     return ActivityWizard(ActivityWizardState(**(data or {})))
    return None

def chatbot_fn(
    message: Optional[str],
    history: List[ChatMessage],
    state: Optional[Dict[str, Any]]
) -> Tuple[List[ChatMessage], Optional[Dict[str, Any]], str]:
    """
    Core chatbot function driving the form-filling dialogue.

    Args:
        message: The latest user input (or None on initial load).
        history: List of (user_msg, bot_msg) tuples for display.
        state: A dict holding dialogue state, or None to initialize. Gradio calls the chatbot_fn on each user input, therefore state is a persitent memory.

    Returns:
        A tuple of (updated history, updated state).
    """
    # --- Initialization on first call ---
    if state is None:
        state = {
            "form_type": None,   # which form the user has chosen
            "lang": None,        # language code for future multi-language support
            "responses": {},     # stores slot_name -> user_response
            "idx": 0,            # pointer into the slots list
            "pdf_file": None,     # path to the pdf file for later overwrite
            "active_wizzard":None, # whether there is currently a wizzard active 
            "wizzard_handles":None # handles object of the current wizzard
        }
    # safe user message
    if message is not None:
        history.append(ChatMessage(role="user", content=message))

    # -------------------------
    # Wizard-Router (immer VOR dem restlichen Flow)
    # -------------------------
    while True:
        active_name = state.get("active_wizard")
        wizard_state_data = state.get("wizard_state")
        wizard = build_wizard_from_state(active_name, wizard_state_data) if active_name else None

        # Falls kein aktiver Wizard, prüfen ob einer gestartet werden muss
        if not wizard:
            if state.get("lang") is None:
                wizard = LanguageWizard()
                state["active_wizard"] = "language_wizard"
            elif state.get("form_type") is None:
                wizard = FormSelectionWizard(
                    FormSelectionWizardState(
                        lang_code=state.get("lang"),
                        available_form_keys=sorted(list(FORMS.keys()))
                    )
                )
                state["active_wizard"] = "form_selection_wizard"
            else:
                # # >>> NEU: Activity-Wizard auto-starten, wenn aktueller Slot 'activity' ist
                # form_key = state.get("form_type")
                # if form_key:
                #     slots_def = FORMS[form_key]["slots"]
                #     cur_idx, _ = next_slot_index(slots_def, state)
                #     if cur_idx is not None:
                #         slot_def = slots_def[cur_idx]
                #         slot_name = slot_def.get("slot_name")
                #         # Option A: fester Slotname
                #         if slot_name == "activity" or slot_def.get("use_activity_wizard") is True:
                #             wizard = ActivityWizard(
                #                 ActivityWizardState(lang_code=state.get("lang"))
                #             )
                #             state["active_wizard"] = "activity_wizard"
                #         else:
                #             # Kein Wizard nötig → AUS DER SCHLEIFE RAUS
                #             break
                #     else:
                #         # Kein Wizard nötig → AUS DER SCHLEIFE RAUS
                #         break
                break

        # Wenn Wizard frisch gestartet wurde (turns == 0): step(None) → Initialprompt
        # Wenn Wizard schon läuft: step(message) → verarbeitet Nutzereingab
        user_text = message
        reply, done, lang_code = wizard.step(user_text)
        history = utter_message_with_translation(history=history, prompt=reply, target_lang = state.get('lang'), source_lang = lang_code)
        state["wizard_state"] = wizard.export_state()

        if done:
            # Nebenwirkungen übertragen
            if state["active_wizard"] == "language_wizard":
                state["lang"] = state["wizard_state"].get("lang_code")

                # instruction message
                instruction_msg = instruction_msgs.get(state["lang"], instruction_msgs["de"])
                history = utter_message_with_translation(history, instruction_msg, state.get('lang'))
            elif state["active_wizard"] == "form_selection_wizard":
                selected = state["wizard_state"].get("selected_form_key")
                if selected:
                    state["form_type"] = selected
                    state["idx"] = 0
                    state["pdf_file"] = FORMS[selected]["pdf_file"]
                    state["awaiting_first_slot_prompt"] = True
            elif state["active_wizard"] == "activity_wizard":  # NEU
                # Ergebnis in responses einsetzen und Slot weiterschalten
                final_text = state["wizard_state"].get("final_activity_text")
                if final_text:
                    # Wir müssen wissen, welcher Slot gerade dran war:
                    slots_def = FORMS[state["form_type"]]["slots"]
                    cur_idx, _ = next_slot_index(slots_def, state)
                    if cur_idx is not None:
                        slot_def = slots_def[cur_idx]
                        if slot_def.get("slot_name") == "activity":
                            target_filed_name = slot_def.get("filed_name")
                            state["responses"]["activity"] = {
                                "value": final_text,
                                "target_filed_name": target_filed_name
                            }
                            # current slot erledigt → Index erhöhen
                            state["idx"] = cur_idx + 1
            # Wizard schließen
            state["active_wizard"] = None
            state["wizard_state"] = None
            continue
        else:
            # Wizard wartet auf Nutzerantwort → Turn HIER beenden, restlicher Bot-Flow pausiert
            return history, state, ""
        
    # --- Gate: erste Slot-Frage nach Formularwahl ---
    if state.get("awaiting_first_slot_prompt"):
        state["awaiting_first_slot_prompt"] = False

        # Slot 0 ermitteln
        slots = FORMS[state["form_type"]]["slots"]
        first_slot = slots[0]

        state["show_upload"] = bool(first_slot.get("show_upload", False))
        state["upload_label"] = first_slot.get("upload_label", "Dateien hochladen")
        # translate label if nessecary
        if state.get('lang') and state.get('lang') != 'de':
            state["upload_label"] = translate_from_de(state["upload_label"], state.get('lang'))

        # # Prompt bauen (ggf. später lokalisieren)
        # prompt = first_slot.get("prompt", first_slot.get("description", ""))

        # # Choice-Optionen anfügen (nummeriert)
        # if first_slot.get("slot_type") == "choice":
        #     options = first_slot["choices"]
        #     prompt += "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        prompt = compose_prompt_for_slot(first_slot)

        # Bot-Ausgabe + Turn hier beenden, damit 'message' nicht als Slot-Antwort verarbeitet wird
        history = utter_message_with_translation(history, prompt, state.get('lang'))
        return history, state, ""

    # --- Classify Edit intent via LLM classification ---
    msg_low = (message or "").lower()
    if state.get("form_type") and message and any(cmd in msg_low for cmd in EDIT_CMDS[state.get("lang")]):
        # Slot-Beschreibungen sammeln
        slots_def = FORMS[state["form_type"]]["slots"]
        descriptions = "\n".join(
            f"{slot_def['slot_name']}: {slot_def.get('description','')}" 
            for slot_def in slots_def
        )
        # Prompt für LLM
        classify_prompt = (
            f"Basierend auf der Nutzeranfrage: '{message}' und den folgenden Slot-Beschreibungen,"
            " gib nur den Slot-Namen zurück, der geändert werden soll, ohne zusätzliche Erklärungen:\n" + descriptions
        )
        slot_key = LLMValidatorService.validate_openai(classify_prompt, "gpt-4.1-mini", OpenAI())
        if slot_key and any(slot_def['slot_name'] == slot_key for slot_def in slots_def):
            # Index des Slots finden
            edit_idx = next(
                (i for i, slot_def in enumerate(slots_def) if slot_def['slot_name'] == slot_key),
                None
            )
            if edit_idx is not None:
                # Resume-Index merken
                state['resume_idx'] = state['idx']
                state['edit_idx'] = edit_idx
                # Nur den gewünschten Slot löschen
                state['responses'].pop(slot_key, None)
                # Auf den zu bearbeitenden Slot springen
                state['idx'] = edit_idx
                # Prompt für diesen Slot neu stellen
                prompt = FORMS[state['form_type']]['slots'][edit_idx].get('prompt', FORMS[state['form_type']]['slots'][edit_idx].get('description', ''))
                # Falls Choice-Slot, Optionen anhängen
                slot_def = slots_def[edit_idx]
                # if slot_def['slot_type'] == 'choice':
                #     opts = slot_def['choices']
                #     prompt += "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
                prompt = compose_prompt_for_slot(slot_def)
                history = utter_message_with_translation(history = history, prompt=prompt, target_lang = state.get('lang'))
                return history, state, ""
        # Wenn Klassifikation fehlschlägt, weiter normal

    # --- Step 2: Handle input for current slot ---
    # here we already have a from selected
    form_conf  = FORMS[state["form_type"]] # get the selected formtype from state
    slots_def  = form_conf["slots"] # get the slots from the selecetd form
    validators = form_conf["validators"] # get the respective validators from the from 
    # get next index
    cur_idx, state = next_slot_index(slots_def, state)

    if message is not None and cur_idx is not None:
        # get the info of the current slot
        slot_def   = slots_def[cur_idx]
        slot_name = slot_def["slot_name"]
        slot_type  = slot_def["slot_type"]
        target_filed_name = slot_def.get("filed_name",None)
        check_box_condition = slot_def.get("check_box_condition",None)
        hints = slot_def.get("hints",None)
        # CHOICE slot handling
        if slot_type== "choice":
            # only translate the message if it is not a digit (index)
            if state.get('lang') and state.get('lang') != 'de' and not message.strip().rstrip(".").isdigit():
                message = translate_to_de(message, state.get('lang'))
            selected_choice, matched = valid_choice_slot(message, slot_def, cutoff=0.75)
            if not matched:
                # re-prompt with options if invalid
                opts = slot_def["choices"]
                opt_text = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
                history = utter_message_with_translation(history, f"Ungültige Auswahl. Bitte wählen:\n{opt_text}", state.get('lang'))
                return history, state, ""
            # map to canonical choice
            selection = None

            # interpret user input as index
            if message.strip().rstrip(".").isdigit():
                number = message.strip().rstrip(".")
                selection = slot_def["choices"][int(number)-1]
            else:
            # interpret user input as text
                # for o in slot_def["choices"]:
                #     if message.strip().lower() == o.lower():
                #         selection = o
                #         break
                selection = selected_choice
            

            # If yes/no filed, map to true/false to make it more independent against language
            choices = slot_def["choices"]
            lower_choices = {opt.lower() for opt in choices}
            selection_lc = selection.lower()
            # ja and nein are mapped to true false regardless of which other choices are present
            if {"ja", "nein"} & lower_choices and selection_lc in {"ja", "nein"}:
                value = map_yes_no_to_bool(selection)
            else:
                value = selection
            # for choices regarding boxes
            if check_box_condition:
                state["responses"][slot_name] = {"value" : value, "target_filed_name": target_filed_name, "choices": slot_def['choices'], "check_box_condition":check_box_condition}
            # if there is no associated checkbox in the document
            if isinstance(target_filed_name, str):
                state["responses"][slot_name] = {"value" : value, "target_filed_name": target_filed_name}
            else:
                state["responses"][slot_name] = {"value" : value, "target_filed_name": target_filed_name, "choices": slot_def.get("choices")}

            # show hints if there are any
            if hints:
                # are there hints for the current input
                if value in hints:
                    history = utter_message_with_translation(history, hints[value], state.get('lang'))



        # TEXT slot handling
        elif slot_type== "text":
            fn = getattr(validators, f"valid_{slot_name}", BaseValidators.valid_basic) # dynamically get a validator for the current filed if there is one

            # translate to german if necessary
            if state.get('lang') and state.get('lang') != 'de':
                message = translate_to_de(message, state.get('lang'))

            valid, reason, payload = fn(message)
            if not valid: # if input is not valid
                history = utter_message_with_translation(history, f"Ungültige Eingabe.\n{reason}\nBitte versuche es nocheinmal.", state.get('lang'))
                return history, state, ""
            elif valid and reason != "":
                history = utter_message_with_translation(history, reason, state.get('lang'))
            # if input is valid
            state["responses"][slot_name] = {"value" : payload, "target_filed_name": target_filed_name}

            # show hints if there are any
            if hints:
                # are there hints for the current input
                if message in hints:
                    history = utter_message_with_translation(history, hints[message], state.get('lang'))

        # advance to next
        state["idx"] = cur_idx + 1

        # —————————————————————————————————————————————
        # If we have just come out of a “change slot” flow, then jump
        # back to the original position:
        if state.get("edit_idx") is not None and cur_idx == state["edit_idx"]:
            # get original position and delete flag
            resume = state.pop("resume_idx")
            state.pop("edit_idx", None)
            # continue
            state["idx"] = resume
        # —————————————————————————————————————————————

    # --- Step 3: Ask next question or finish ---
    next_idx, state = next_slot_index(slots_def, state)
    if next_idx is not None:
        next_slot_def   = slots_def[next_idx]

        state["show_upload"] = bool(next_slot_def.get("show_upload", False))
        state["upload_label"] = next_slot_def.get("upload_label", "Dateien hochladen")
        state["uploaded_files"] = None
        # translate label if nessecary
        if state.get('lang') and state.get('lang') != 'de':
            state["upload_label"] = translate_from_de(state["upload_label"], state.get('lang'))

        next_slot_name = next_slot_def["slot_name"]
        prompt = next_slot_def.get('prompt', next_slot_def.get('description', ''))
        # if next_slot_def["slot_type"] == "choice":
        #     opts = next_slot_def["choices"]
        #     opt_lines = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(opts))
        #     prompt += "\n" + opt_lines
        prompt = compose_prompt_for_slot(next_slot_def)
        history = utter_message_with_translation(history,prompt,state.get('lang'))
    else:
        history = utter_message_with_translation(
            history,
            "Vielen Dank! Das Formular ist abgeschlossen. Nachdem Sie das Formular unterschrieben haben, können Sie es hier zur elektronischen Übermittlung direkt hochladen",
            state.get('lang')
        )
        print_summary(state=state, forms=FORMS)
        state['completed'] = True                      # Download-Button jetzt sichtbar
        state['awaiting_final_upload'] = True          # <<< NEU: Nächster Upload ist der finale
        state['show_upload'] = True                    # Upload-Button einblenden
        state['upload_label'] = (
            'Unterschriebenes Formular hochladen und Vorgang abschließen.'
            if state.get('lang', 'de') == 'de' else
            translate_from_de('Unterschriebenes Formular hochladen und Vorgang abschließen.', state.get('lang'))
        )
        state["uploaded_files"] = None                 # Anzeige leeren
    return history, state, ""


