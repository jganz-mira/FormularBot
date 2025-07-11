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
    responses: Dict[str, str],
    start_idx: int
) -> Optional[int]:
    """
    Find the index of the next slot to ask, skipping those whose
    'condition' (if any) is not met.

    Args:
        slots_def: List of slot definition dicts.
        responses: Dict mapping slot_name to previously given answers.
        start_idx: The index to start searching from.

    Returns:
        The index of the next askable slot, or None if all slots are done.
    """
    for i in range(start_idx, len(slots_def)):
        slot_def = slots_def[i]
        cond = slot_def.get("condition")
        # if a condition is defined, skip unless it's fulfilled
        if cond:
            prev_val = responses.get(cond["slot_name"])
            ###debug####
            print(cond["slot_value"])
            if prev_val['value'] != cond["slot_value"]:
                continue
        return i
    return None

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
        label = form_conf["prompt_map"][lang].get(slot_name, slot_name)
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
        # Nur Slot eintragen, wenn mindestens value vorhanden ist
        if value is not None:
            out_data[slot_name] = {
                "value": value,
                "target_filed_name": target,
                "choices": choices
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
