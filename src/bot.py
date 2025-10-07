# bot.py — Vereinheitlicht & klar strukturierter Wizard-Ablauf
# - Nur noch 'active_wizard' (ein z): 'language_wizard' | 'form_selection_wizard' | 'shortcut_wizard'
# - Nur noch 'wizard_handles' (ein z) für optionale UI-Handles
# - Sichtbar markierte Abschnitte: START / END / NEXT START je Wizard

from typing import Optional, Tuple, List, Dict, Any
import os
from openai import OpenAI
from datetime import date, timedelta, datetime

from .validators import BaseValidators, GewerbeanmeldungValidators
from .bot_helper import (
    load_forms, next_slot_index, print_summary, map_yes_no_to_bool,
    save_responses_to_json, utter_message_with_translation,
    compose_prompt_for_slot, valid_choice_slot
)
from .wizards import (
    LanguageWizard, LanguageWizardState,
    FormSelectionWizard, FormSelectionWizardState
)
from .translator import translate_from_de, translate_to_de
from gradio import ChatMessage

# ---------------------------------------------------------------------------
# Projekt-Setup: Formulare laden (z. B. debug_zwei.json im erwarteten Format)
# ---------------------------------------------------------------------------
base_path   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forms_path  = os.path.join(base_path, 'forms', 'ge')

validator_map = {
    "BaseValidators": BaseValidators,
    "GewerbeanmeldungValidators": GewerbeanmeldungValidators(),
}

FORMS = load_forms(form_path=forms_path, validator_map=validator_map)

# ---------------------------------------------------------------------------
# Helfer für History-Kompatibilität (tuple | dict | ChatMessage)
# ---------------------------------------------------------------------------
def _role_of(msg: Any) -> Optional[str]:
    if isinstance(msg, ChatMessage):
        return msg.role
    if isinstance(msg, dict):
        return msg.get("role")
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        return msg[0]
    return None

def _content_of(msg: Any) -> str:
    if isinstance(msg, ChatMessage):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        return str(msg[1])
    return ""

def _append_user_once(history: List[Any], user_text: Optional[str]) -> List[Any]:
    """
    Verhindert doppelte User-Einträge (Streamlit schreibt i. d. R. bereits in die History).
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return history
    if history and _role_of(history[-1]) == "user" and _content_of(history[-1]).strip() == user_text:
        return history
    history.append(ChatMessage(role="user", content=user_text))
    return history

# ---------------------------------------------------------------------------
# Kernfunktion: linearer Flow gemäß Anforderung
# ---------------------------------------------------------------------------
def chatbot_fn(
    message: Optional[str],
    history: List[Any],
    state: Optional[Dict[str, Any]]
) -> Tuple[List[Any], Optional[Dict[str, Any]], str]:
    """
    Ablauf:
      1) Begrüßung (einmalig)
      2) Sprache abfragen (LanguageWizard)
      3) Formular wählen (FormSelectionWizard)
      4) ShortCutWizard starten (UI liegt in Streamlit)
      5) Nach Mapping: erste offene Slotfrage stellen
      6) Slot-Antwort verarbeiten → nächste Slotfrage
      7) Nach letztem Slot: PDF-Download + Upload-Button aktivieren
    """

    # -----------------------------------------------------------------------
    # 0) State initialisieren — saubere, klare Keys
    # -----------------------------------------------------------------------
    if state is None:
        state = {
            "form_type": None,            # Key des gewählten Formulars
            "lang": None,                 # ISO-639-1 Sprachcode
            "responses": {},              # slot_name -> {"value":..., "target_filed_name":...}
            "idx": 0,                     # Zeiger auf den nächsten zu fragenden Slot
            "pdf_file": None,             # Ziel-PDF
            "active_wizard": None,        # 'language_wizard' | 'form_selection_wizard' | 'shortcut_wizard'
            "wizard_state": None,         # serialisierter Substate des aktiven Wizards
            "wizard_handles": None,       # optionale UI-Handles (Shortcut-Wizard)
            "_greeted": False,            # interne Flag: Begrüßung erfolgt
            "ui": None,                   # UI-Direktive für Streamlit (aktueller Slot)
            "completed": False,           # Abschlussstatus (PDF-Buttons)
            "show_upload": False,         # Upload einblenden
            "upload_label": "Dateien hochladen",
            "awaiting_first_slot_prompt": False  # nach Shortcut-Mapping: erste Slotfrage
        }

    # Nutzer-Text ggf. einmalig in History übernehmen (keine Duplikate)
    history = _append_user_once(history, message)

    # -----------------------------------------------------------------------
    # 1) Begrüßung (einmalig) — robust ggü. UI-Vorgrüßung
    # -----------------------------------------------------------------------
    if not state.get("_greeted", False):
        greet_de = "👋 Willkommen! Ich helfe Ihnen beim Ausfüllen von Formularen. Los geht’s!"
        history = utter_message_with_translation(history, greet_de, target_lang=state.get("lang"))
        state["_greeted"] = True
        # Kein return: wir gehen direkt zur Sprachauswahl.

    # -----------------------------------------------------------------------
    # 2) Wizard-Router: ggf. nächste Wizard-Stufe starten
    # -----------------------------------------------------------------------
    def _ensure_wizard_started():
        if state.get("active_wizard"):
            return  # bereits aktiv
        if state.get("lang") is None:
            state["active_wizard"] = "language_wizard"
            state["wizard_state"] = None
            return
        if state.get("form_type") is None:
            state["active_wizard"] = "form_selection_wizard"
            state["wizard_state"] = None
            return

    _ensure_wizard_started()

    # -----------------------------------------------------------------------
    # LANGUAGE WIZARD — START
    # -----------------------------------------------------------------------
    if state.get("active_wizard") == "language_wizard":
        wiz = LanguageWizard(LanguageWizardState(**(state.get("wizard_state") or {})))
        user_text = None if state.get("wizard_state") is None else (message or "")
        reply, done, lang_code = wiz.step(user_text)
        history = utter_message_with_translation(history, reply, target_lang=state.get("lang"), source_lang=lang_code)
        state["wizard_state"] = wiz.export_state()

        if not done:
            # wartet auf Nutzereingabe
            return history, state, ""

        # LANGUAGE WIZARD — END
        state["lang"] = state["wizard_state"].get("lang_code") or state.get("lang") or "de"
        state["active_wizard"] = None
        state["wizard_state"] = None

        # LANGUAGE WIZARD — NEXT START: FORM SELECTION WIZARD
        state["active_wizard"] = "form_selection_wizard"
        init_fs_state = FormSelectionWizardState(
            lang_code=state["lang"],
            available_form_keys=sorted(list(FORMS.keys()))
        )
        fs_wiz = FormSelectionWizard(init_fs_state)
        fs_reply, fs_done, fs_lang = fs_wiz.step(None)  # erster Turn ohne Nutzereingabe
        history = utter_message_with_translation(history, fs_reply, target_lang=state.get("lang"), source_lang=fs_lang)
        state["wizard_state"] = fs_wiz.export_state()

        if not fs_done:
            # wartet auf Formularauswahl
            return history, state, ""

    # -----------------------------------------------------------------------
    # FORM SELECTION WIZARD — START (Fortführung & Abschluss)
    # -----------------------------------------------------------------------
    if state.get("active_wizard") == "form_selection_wizard":
        fs_state = state.get("wizard_state") or {}
        wiz = FormSelectionWizard(FormSelectionWizardState(**fs_state))

        # Sicherheit: Sprache und Formularliste setzen
        if wiz.state.lang_code is None:
            wiz.state.lang_code = state.get("lang") or "de"
        if not wiz.state.available_form_keys:
            wiz.state.available_form_keys = sorted(list(FORMS.keys()))

        # Wenn frisch gestartet → erster Turn ohne Nutzereingabe
        user_text = None if not fs_state else (message or "")

        reply, done, fs_lang = wiz.step(user_text)
        history = utter_message_with_translation(history, reply, target_lang=state.get("lang"), source_lang=fs_lang)
        state["wizard_state"] = wiz.export_state()

        if not done:
            # wartet auf Nutzerauswahl
            return history, state, ""

        # FORM SELECTION WIZARD — END
        selected_key = wiz.state.selected_form_key
        if selected_key:
            # Formularwahl übernehmen & persistieren
            state["form_type"] = selected_key
            state["idx"] = 0
            state["pdf_file"] = FORMS[selected_key]["pdf_file"]

            # FORM SELECTION WIZARD — NEXT START: SHORTCUT WIZARD (UI)
            # (UI rendert die 4 Buttons; Bot wartet, bis Mapping abgeschlossen ist.)
            state["active_wizard"] = "shortcut_wizard"
            state["wizard_handles"] = None

            # jetzt ist die UI dran
            return history, state, ""

        # Falls kein Formular gewählt wurde (Edge-Case), Wizard schließen
        state["active_wizard"] = None
        state["wizard_state"] = None
        return history, state, ""

    # -----------------------------------------------------------------------
    # SHORTCUT WIZARD (UI-geführt) — START
    # (Solange aktiv, nichts im Chat ausgeben; UI arbeitet.
    #  Nach dem Finish muss die UI setzen:
    #   - state["active_wizard"] = None
    #   - state["wizard_handles"] = None
    #   - state["awaiting_first_slot_prompt"] = True)
    # -----------------------------------------------------------------------
    if state.get("active_wizard") == "shortcut_wizard":
        return history, state, ""

    # SHORTCUT WIZARD — END → wir erwarten awaiting_first_slot_prompt = True
    if state.pop("awaiting_first_slot_prompt", False):
        slots_def = FORMS[state["form_type"]]["slots"]

        # Sicherheit: von vorne suchen
        if not isinstance(state.get("idx"), int):
            state["idx"] = 0
        else:
            state["idx"] = 0

        # Nächsten wirklich offenen Slot bestimmen
        next_idx, state = next_slot_index(slots_def, state)
        if next_idx is None:
            # Nichts mehr zu tun → direkt in Abschluss-Flow springen
            thanks = (
                "Vielen Dank! Das Formular ist abgeschlossen. "
                "Nachdem Sie das Formular unterschrieben haben, können Sie es hier zur elektronischen Übermittlung direkt hochladen."
            )
            history = utter_message_with_translation(history, thanks, state.get("lang"))
            print_summary(state=state, forms=FORMS)
            state["completed"] = True
            state["awaiting_final_upload"] = True
            state["show_upload"] = True
            base_label_de = "Unterschriebenes Formular hochladen und Vorgang abschließen."
            state["upload_label"] = base_label_de if (state.get("lang") == "de" or not state.get("lang")) \
                else translate_from_de(base_label_de, state["lang"])
            state["uploaded_files"] = None
            return history, state, ""

        next_def = slots_def[next_idx]

        # Upload-Steuerung für die UI (falls Slot Upload vorsieht)
        state["show_upload"]  = bool(next_def.get("show_upload", False))
        state["upload_label"] = next_def.get("upload_label", "Dateien hochladen")
        if state.get("lang") and state["lang"] != "de":
            state["upload_label"] = translate_from_de(state["upload_label"], state["lang"])

        # Prompt + UI-Direktive an die Oberfläche geben
        prompt_text = compose_prompt_for_slot(next_def)
        history = utter_message_with_translation(history, prompt_text, state.get("lang"))
        state["ui"] = _build_ui_for_slot(next_def, state=state)
        return history, state, ""

    # -----------------------------------------------------------------------
    # Falls noch kein Formular gewählt (z. B. initiales Laden), hier enden
    # -----------------------------------------------------------------------
    if not state.get("form_type"):
        return history, state, ""

    # -----------------------------------------------------------------------
    # 6) Aktuellen Slot verarbeiten (Choice/Text)
    # -----------------------------------------------------------------------
    form_conf   = FORMS[state["form_type"]]
    slots_def   = form_conf["slots"]
    validators  = form_conf["validators"]

    cur_idx, state = next_slot_index(slots_def, state)
    if message is not None and cur_idx is not None:
        slot_def   = slots_def[cur_idx]
        slot_name  = slot_def["slot_name"]
        slot_type  = slot_def["slot_type"]
        target     = slot_def.get("filed_name")
        hints      = slot_def.get("hints")
        check_cond = slot_def.get("check_box_condition")

        # --- Choice-Slot -----------------------------------------------------
        if slot_type == "choice":
            # Fremdsprache → nach Deutsch übersetzen, außer Nutzer gibt Index
            user_text = (message or "").strip()
            if state.get("lang") and state["lang"] != "de" and not user_text.rstrip(".").isdigit():
                user_text = translate_to_de(user_text, state["lang"])
            selection, matched = valid_choice_slot(user_text, slot_def, cutoff=0.75)
            if not matched:
                history = utter_message_with_translation(
                    history,
                    "Ungültige Auswahl. Bitte nutzen Sie die Buttons oder geben Sie die Nummer ein.",
                    state.get("lang")
                )
                return history, state, ""
            # Index-Shortcuts unterstützen
            if user_text.rstrip(".").isdigit():
                selection = slot_def["choices"][int(user_text.rstrip(".")) - 1]

            # Ja/Nein vereinheitlichen (true/false als String)
            lc = (selection or "").strip().lower()
            if lc in {"ja", "nein"}:
                value = map_yes_no_to_bool(selection)
            else:
                value = selection

            payload = {"value": value, "target_filed_name": target}
            if "choices" in slot_def:
                payload["choices"] = slot_def["choices"]
            if check_cond:
                payload["check_box_condition"] = check_cond
            state["responses"][slot_name] = payload

            # Hinweise dynamisch ausspielen
            if hints and selection in hints:
                history = utter_message_with_translation(history, hints[selection], state.get("lang"))

        # --- Text-Slot -------------------------------------------------------
        elif slot_type == "text" or slot_type == "date":
            # ggf. nach Deutsch übersetzen
            user_text = message or ""
            if state.get("lang") and state["lang"] != "de":
                user_text = translate_to_de(user_text, state["lang"])

            # Feldspezifische Validierung (Fallback: Basic)
            validate_fn = getattr(validators, f"valid_{slot_name}", BaseValidators.valid_basic)
            is_valid, reason, normalized_value = validate_fn(user_text)
            if not is_valid:
                history = utter_message_with_translation(
                    history, f"Ungültige Eingabe.\n{reason}\nBitte versuche es erneut.", state.get("lang")
                )
                return history, state, ""
            if reason:  # optionale Info aus Validator
                history = utter_message_with_translation(history, reason, state.get("lang"))

            state["responses"][slot_name] = {"value": normalized_value, "target_filed_name": target}

        # Slot abgeschlossen → UI-Direktive leeren und Index erhöhen
        state.pop("ui", None)
        state["idx"] = cur_idx + 1

    # -----------------------------------------------------------------------
    # 7) Nächsten Slot fragen ODER Abschluss einleiten
    # -----------------------------------------------------------------------
    next_idx, state = next_slot_index(slots_def, state)
    if next_idx is not None:
        next_def = slots_def[next_idx]

        # Upload-Steuerung für die UI (falls Slot Upload vorsieht)
        state["show_upload"]  = bool(next_def.get("show_upload", False))
        state["upload_label"] = next_def.get("upload_label", "Dateien hochladen")
        state["uploaded_files"] = None
        if state.get("lang") and state["lang"] != "de":
            state["upload_label"] = translate_from_de(state["upload_label"], state["lang"])

        # Prompt + UI-Direktive an die Oberfläche geben
        prompt_text = compose_prompt_for_slot(next_def)
        history = utter_message_with_translation(history, prompt_text, state.get("lang"))
        state["ui"] = _build_ui_for_slot(next_def, state=state)
        return history, state, ""

    # --- Alle Slots fertig → Abschlussbotschaft, PDF/Upload signalisieren ---
    thanks = (
        "Vielen Dank! Das Formular ist abgeschlossen. "
        "Nachdem Sie das Formular unterschrieben haben, können Sie es hier zur elektronischen Übermittlung direkt hochladen."
    )
    history = utter_message_with_translation(history, thanks, state.get("lang"))
    print_summary(state=state, forms=FORMS)
    state["completed"] = True
    state["awaiting_final_upload"] = True
    state["show_upload"] = True
    base_label_de = "Unterschriebenes Formular hochladen und Vorgang abschließen."
    state["upload_label"] = base_label_de if (state.get("lang") == "de" or not state.get("lang")) else translate_from_de(base_label_de, state["lang"])
    state["uploaded_files"] = None

    return history, state, ""

# ---------------------------------------------------------------------------
# UI-Beschreibung eines Slots → für Streamlit (Radio/Text + Zusatzinfos)
# ---------------------------------------------------------------------------
def _parse_ddmmyyyy_to_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except Exception:
        return None
    
def _coerce_str_date(value: str | date | None) -> str:
    """Bringt einen Wert in 'TT.MM.JJJJ'-Stringform (für PDF/State)."""
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, str) and value.strip():
        # akzeptiere bereits TT.MM.JJJJ
        d = _parse_ddmmyyyy_to_date(value)
        if d:
            return d.strftime("%d.%m.%Y")
        # akzeptiere ISO 'YYYY-MM-DD'
        try:
            d = datetime.strptime(value, "%Y-%m-%d").date()
            return d.strftime("%d.%m.%Y")
        except Exception:
            pass
    return ""
    
def _build_ui_for_slot(slot_def: Dict[str, Any], state) -> Dict[str, Any]:
    ui: Dict[str, Any] = {
        "slot_name": slot_def.get("slot_name"),
        "component": None,  # "radio" | "text_input" | ...
        "args": {},
        "additional_information": slot_def.get("additional_information", []),
        "send_on_change": False,
        "slot_description": slot_def.get("description", "")
    }
    stype = slot_def.get("slot_type")
    if stype == "choice":
        ui["component"] = "radio"
        ui["args"] = {"label": "Bitte auswählen:", "options": slot_def.get("choices", [])}
    elif stype == "text":
        ui["component"] = "text_input"
        ui["args"] = {
            "label": slot_def.get("ui_label", "Antwort eingeben"),
            "placeholder": slot_def.get("placeholder", "")
        }
    elif stype == "date":
        # bestehender Wert (falls vorhanden) in date für das Widget umwandeln
        existing = (state.get("responses", {}).get(slot_def["slot_name"], {}) or {}).get("value")
        # existing kann str "TT.MM.JJJJ" / "YYYY-MM-DD" oder None sein
        existing_date = _parse_ddmmyyyy_to_date(existing) or (
            datetime.strptime(existing, "%Y-%m-%d").date() if isinstance(existing, str) and len(existing) == 10 and existing[4] == "-" else None
        )

        # optionale Grenzen aus JSON
        constraints = slot_def.get("constraints") or {}
        max_past = int(constraints.get("max_offset_days_past", 3650))  # default 10 Jahre
        max_future = int(constraints.get("max_offset_days_future", 3650))

        today = date.today()
        min_value = today - timedelta(days=max_past) if max_past >= 0 else None
        max_value = today + timedelta(days=max_future) if max_future >= 0 else None

        ui = {
            "slot_name": slot_def["slot_name"],
            "component": "date_input",
            "args": {
                "label": "Datum",
                # value darf None sein (leeres Widget), sonst date
                "value": existing_date,         # Streamlit nimmt date|None
                "min_value": min_value,
                "max_value": max_value,
                "help": slot_def.get("description", ""),
            },
            "slot_description": slot_def.get("description", ""),
            "additional_information": slot_def.get("additional_information", []),
        }
    return ui
