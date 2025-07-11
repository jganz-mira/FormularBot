
from typing import Optional, Tuple, List, Dict, Any
import os
from .validators import BaseValidators, GewerbeanmeldungValidators
from .bot_helper import load_forms, next_slot_index, print_summary, map_yes_no_to_bool, save_responses_to_json
from .llm_validator_service import LLMValidatorService
from openai import OpenAI
from gradio import ChatMessage


validator_map = {
    "BaseValidators": BaseValidators,
    "GewerbeanmeldungValidators": GewerbeanmeldungValidators,
}

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

FORMS = load_forms(
    form_path = forms_path,
    validator_map = validator_map
)

EDIT_CMDS = {"ändern", "korrigieren", "korrektur", "update"} # diese kommen später an einen anderen Ort

# Assume FORMS is a dict loaded at startup mapping form keys to their JSON configs,
# e.g.:
# from src.form_loader import FORMS

def valid_choice_slot(message: str, slot_def: Dict[str, Any]) -> bool:
    """
    Validate a 'choice' slot by checking if the user's input corresponds
    to one of the defined choices.

    - If the input is a digit, interpret it as a 1-based index into 'choices'.
    - Otherwise, compare (case-insensitive) to each choice string.

    Args:
        message: The raw user input.
        slot_def: The slot definition dict, must contain a "choices": List[str].

    Returns:
        True if the input maps to one of the choices, False otherwise.
    """
    choices: List[str] = slot_def.get("choices", [])
    text = message.strip()

    # 1) Digit input as index?
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(choices):
            return True

    # 2) Exact text match (case-insensitive)
    for opt in choices:
        if text.lower() == opt.lower():
            return True

    return False

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
            "lang": "de",        # language code for future multi-language support
            "responses": {},     # stores slot_name -> user_response
            "idx": 0,            # pointer into the slots list
            "pdf_file": None     # path to the pdf file for later overwrite
        }

    print(state)

    msg_low = (message or "").lower()
    # --- Classify Edit intent via LLM classification ---
    if state.get("form_type") and message and any(cmd in msg_low for cmd in EDIT_CMDS):
        history.append(ChatMessage(role='user',content=message))
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
                prompt = FORMS[state['form_type']]['prompt_map'][state['lang']][slot_key]
                # Falls Choice-Slot, Optionen anhängen
                slot_def = slots_def[edit_idx]
                if slot_def['slot_type'] == 'choice':
                    opts = slot_def['choices']
                    prompt += "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
                history.append(ChatMessage(role='assistant',content=prompt))
                return history, state, ""
        # Wenn Klassifikation fehlschlägt, weiter normal


    # --- Step 1: Form selection ---
    if state["form_type"] is None: # if no form has been selected yet
        available = list(FORMS.keys())
        if message and message in available: # does the user input exactly match one of the listed forms?
            # user selected a form
            state["form_type"] = message
            state["idx"] = 0
            history.append(ChatMessage(role="user", content=message)) # store user input in history
            history.append(ChatMessage(role="assistant", content=f"Sie haben das Formular **{message}** gewählt.")) # store bot output in history
            # immediately ask first slot
            first_idx = next_slot_index(FORMS[message]["slots"], state["responses"], 0)
            # slect the slots based on the chosen form, slot0_def contains the complete definition of the first slot
            slot0_def = FORMS[message]["slots"][first_idx]
            # get the prompt for the respective slot
            prompt0 = FORMS[message]["prompt_map"][state["lang"]][slot0_def["slot_name"]]
            # if slot is of type choice, load available choices and list them
            if slot0_def["slot_type"] == "choice":
                # append enumerated options
                opts = slot0_def["choices"]
                opt_lines = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(opts))
                prompt0 += "\n" + opt_lines
            # save the path to the template pdf file to the state
            state["pdf_file"] = FORMS[message]["pdf_file"]
            # add the fully formatted promt to history
            history.append(ChatMessage(role='assistant',content=prompt0))
            return history, state, ""
        else:
            # ask which form to fill out, if non is chosen already
            form_list = "\n".join(f"- {f}" for f in available)
            history.append(ChatMessage(role='assistant',content=f"Welches Formular möchten Sie ausfüllen?\n{form_list}"))
            return history, state, ""

    # --- Step 2: Handle input for current slot ---
    # here we already have a from selected
    form_conf  = FORMS[state["form_type"]] # get the selected formtype from state
    slots_def  = form_conf["slots"] # get the slots from the selecetd form
    prompts    = form_conf["prompt_map"][state["lang"]] # get the promts the bot will utter from the form
    validators = form_conf["validators"] # get the respective validators from the from 
    # get next index
    cur_idx = next_slot_index(slots_def, state["responses"], state["idx"])

    if message is not None and cur_idx is not None:
        # history.append(ChatMessage(role="user", content=message)) # To show user message in chat history
        # get the info of the current slot
        slot_def   = slots_def[cur_idx]
        slot_name = slot_def["slot_name"]
        slot_type  = slot_def["slot_type"]
        target_filed_name = slot_def.get("filed_name",None)

        # CHOICE slot handling
        if slot_type== "choice":
            if not valid_choice_slot(message, slot_def):
                # re-prompt with options if invalid
                opts = slot_def["choices"]
                opt_text = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
                history.append(ChatMessage(role='user',content=message)) # show waht user typed
                history.append(ChatMessage(role='assistant',content=f"Ungültige Auswahl. Bitte wählen:\n{opt_text}"))
                return history, state, ""
            # map to canonical choice
            selection = None
            if message.isdigit():
                selection = slot_def["choices"][int(message)-1]
            else:
                for o in slot_def["choices"]:
                    if message.strip().lower() == o.lower():
                        selection = o
                        break

            # If yes/no filed, map to true/false to make it more independent against language
            choices = slot_def["choices"]
            if set(opt.lower() for opt in choices) == {"ja", "nein"}:
                value = map_yes_no_to_bool(selection)
            else:
                value = selection
            state["responses"][slot_name] = {"value" : value, "target_filed_name": target_filed_name, "choices": slot_def['choices']}

        # TEXT slot handling
        elif slot_type== "text":
            fn = getattr(validators, f"valid_{slot_name}", None) # dynamically get a validator for the current filed if there is one
            if fn and not fn(message): # if there is a function and the function returned false
                history.append(ChatMessage(role='user',content=message))
                history.append(ChatMessage(role='assistant',content=f"Ungültige Eingabe für **{slot_name}**. Bitte erneut:"))
                return history, state, ""
            state["responses"][slot_name] = {"value" : message, "target_filed_name": target_filed_name}

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
    next_idx = next_slot_index(slots_def, state["responses"], state["idx"])
    if next_idx is not None:
        next_slot_def   = slots_def[next_idx]
        next_slot_name = next_slot_def["slot_name"]
        prompt    = prompts[next_slot_name]
        if next_slot_def["slot_type"] == "choice":
            opts = next_slot_def["choices"]
            opt_lines = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(opts))
            prompt += "\n" + opt_lines
        history.append(ChatMessage(role='assistant',content=prompt))
    else:
        history.append(ChatMessage(role='assistant',content="Vielen Dank! Das Formular ist abgeschlossen."))
        save_responses_to_json(state=state, output_path="out/out.json")
        print_summary(state = state, forms = FORMS)
        state = None

    return history, state, ""

