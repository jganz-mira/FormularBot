"""
streamlit_main.py — Streamlit-UI für den Formular-Chatbot

Ziele dieser Überarbeitung:
- Begrüßung/Sprachauswahl erscheint zuverlässig beim ersten Laden (Start-Turn).
- Klar strukturierter Ablauf mit sprechenden Variablennamen und eindeutigen Kommentaren.
- Dynamische Slot-Widgets (Radio/Text) + Expander mit Zusatzinfos und Mini-Chat (stateful pro Slot).
- Abschluss-Flow: PDF erzeugen, Download anbieten, optionaler Upload-Mockup.

Voraussetzungen:
- Bot-Logik: src.bot.chatbot_fn(history, state) liefert neue History/State zurück
- PDF-Füller: src.pdf_backend.GenericPdfFiller
- Übersetzungen: src.translator.*
- Helper: src.bot_helper.save_responses_to_json
"""

from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd
from io import StringIO
import json
import re
from datetime import date as _date

import streamlit as st
from openai import OpenAI

# --- Projektabhängige Importe ---
from src import bot_helper
from src.bot import chatbot_fn
from src.bot_helper import save_responses_to_json, load_forms
from src.pdf_backend import GenericPdfFiller
from src.translator import final_msgs, download_button_msgs, files_msgs, pdf_file_msgs
from src.wizards import ShortCutWizard, ShortCutWizardState, IDCardWizard, IDCardWizardState
from src.bot_helper import extract_information_HRA_info_from_img, extract_information_id_card
from src.validators import BaseValidators, GewerbeanmeldungValidators

# =============================================================================
# Konstante(n) & Basiskonfiguration
# =============================================================================
PAGE_TITLE = "Göppingen Chatbot"
PAGE_ICON = "💬"
CHAT_INPUT_PLACEHOLDER = "Ihre Nachricht hier eingeben …"
GREETING_TEXT = "👋 Willkommen! Ich helfe Ihnen beim Ausfüllen von Formularen. Los geht’s!"

# base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# forms_path = os.path.join(base_path, 'GPBot_v3/forms', 'ge')
repo_root  = Path(__file__).resolve().parent
forms_path = (repo_root / "forms" / "ge").as_posix() 

validator_map = {
    "BaseValidators": BaseValidators,
    "GewerbeanmeldungValidators": GewerbeanmeldungValidators(),
}

FORMS = load_forms(
        form_path = forms_path,
        validator_map = validator_map
    )


# Seite konfigurieren (einmalig)
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
try:
    # Optional: Nur wenn vorhanden (neuere Streamlit-Versionen)
    st.logo(image="logo-black.png", size="large", link="https://www.mira.vision/")
except Exception:
    pass

st.title("MVP Gewerbeanmeldungen by MIRA Vision")


# =============================================================================
# Debuggingfunktionen
# =============================================================================
def _fmt_target_field_name(x) -> str:
    if isinstance(x, list):
        return ", ".join(map(str, x)) if x else "∅"
    if x in (None, "", []):
        return "∅"
    return str(x)

def debug_print_responses_to_terminal():
    """Slot-Antworten im Terminal (stdout) ausgeben – inkl. locked & nächster offener Slot."""
    state = st.session_state.get("state", {}) or {}
    responses = state.get("responses", {}) or {}

    if not responses:
        print("⚠️  Keine Slot-Antworten im State gefunden.")
        return

    form_key = state.get("form_type", "—")
    lang = state.get("lang", "—")
    idx = state.get("idx", 0)

    print("\n=== DEBUG: Slot-Antworten ===========================")
    print(f"Formular: {form_key} | Sprache: {lang} | idx={idx}")
    print(f"Anzahl Antworten: {len(responses)}")

    # In sinnvoller Reihenfolge ausgeben, falls wir slots_def finden
    slots_def = None
    try:
        forms_in_session = st.session_state.get("FORMS") or {}
        if form_key and forms_in_session:
            slots_def = (forms_in_session.get(form_key) or {}).get("slots")
    except Exception:
        slots_def = None

    ordered_names = (
        [s.get("slot_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")]
        if slots_def else
        sorted(responses.keys())
    )

    import json as _json
    for name in ordered_names:
        entry = responses.get(name, {})
        val = entry.get("value")
        locked = entry.get("locked", False)

        if isinstance(val, (dict, list)):
            val_repr = _json.dumps(val, ensure_ascii=False, separators=(",", ":"))
        elif val in ("", None):
            val_repr = "∅"
        else:
            val_repr = str(val)

        lock_tag = " [locked]" if locked else ""
        print(f"{name} -> {val_repr}{lock_tag}")

    # Optional: nächsten offenen Slot bestimmen
    try:
        if slots_def:
            from bot_helper import next_slot_index  # nutzt deine aktualisierte Logik
            # Kopie des States, falls die Funktion den State modifiziert
            state_copy = dict(state)
            next_idx, _ = next_slot_index(slots_def, state_copy)
            if next_idx is None:
                print("Nächster offener Slot: ✅ keiner (alle fertig).")
            else:
                next_name = slots_def[next_idx]["slot_name"]
                print(f"Nächster offener Slot: #{next_idx} → {next_name}")
    except Exception as e:
        print(f"(Hinweis) next_slot_index nicht aufrufbar: {e}")

    print("=====================================================\n")

def render_debug_panel():
    """Zeigt ein Debug-Panel mit allen bisher gesetzten Slots (inkl. locked, Export & Terminal-Dump)."""
    with st.expander("🔧 Debug-Panel: Bisherige Slot-Antworten", expanded=False):
        state = st.session_state.get("state", {}) or {}
        responses = state.get("responses", {}) or {}

        # Meta-Infos
        col1, col2, col3 = st.columns(3)
        col1.metric("Formular", state.get("form_type", "—"))
        col2.metric("Gesetzte Slots", len(responses))
        col3.metric("Sprache", state.get("lang", "—"))

        # Versuche den nächsten offenen Slot zu bestimmen
        next_slot_label = "—"
        try:
            forms_in_session = st.session_state.get("FORMS") or {}
            form_key = state.get("form_type")
            slots_def = (forms_in_session.get(form_key) or {}).get("slots") if (forms_in_session and form_key) else None
            if slots_def:
                from bot_helper import next_slot_index
                state_copy = dict(state)
                nxt, _ = next_slot_index(slots_def, state_copy)
                next_slot_label = "keiner (alle fertig)" if nxt is None else slots_def[nxt]["slot_name"]
        except Exception:
            pass
        st.caption(f"**Nächster offener Slot:** {next_slot_label}")

        if not responses:
            st.info("Noch keine Slot-Antworten vorhanden.")
            if st.button("🔍 Debug: Slot-Antworten ins Terminal ausgeben"):
                debug_print_responses_to_terminal()
            return

        # In DataFrame bringen
        rows = []
        for slot_name, data in responses.items():
            val = data.get("value")
            # kompakte Darstellung für UI-Tabelle
            if isinstance(val, (dict, list)):
                try:
                    val = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    val = str(val)
            elif val in ("", None):
                val = "∅"

            rows.append({
                "slot_name": slot_name,
                "value": val,
                "target_filed_name":  _fmt_target_field_name(data.get("target_filed_name")),
                "locked": bool(data.get("locked", False)),
                "has_choices": "choices" in data,
            })

        df = pd.DataFrame(rows, columns=["slot_name", "value", "target_filed_name", "locked", "has_choices"])
        st.dataframe(df, use_container_width=True)

        # Downloads (CSV/JSON)
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ CSV exportieren",
            data=csv_buf.getvalue(),
            file_name="slot_responses.csv",
            mime="text/csv"
        )

        json_str = json.dumps(responses, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ JSON exportieren",
            data=json_str.encode("utf-8"),
            file_name="slot_responses.json",
            mime="application/json"
        )

        # Optional: Terminal-Print
        if st.button("🔍 Debug: Slot-Antworten ins Terminal ausgeben"):
            debug_print_responses_to_terminal()


# =============================================================================
# Hilfsfunktionen: History-Kompatibilität & PDF-Erzeugung
# =============================================================================

def role_of(message: Any) -> str | None:
    """Liest die Rolle aus verschiedenen möglichen Message-Formaten."""
    if isinstance(message, dict):
        return message.get("role")
    if hasattr(message, "role"):
        return getattr(message, "role", None)
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return message[0]
    return None


def content_of(message: Any) -> str:
    """Liest den Textinhalt aus verschiedenen möglichen Message-Formaten."""
    if isinstance(message, dict):
        return message.get("content", "")
    if hasattr(message, "content"):
        return getattr(message, "content", "")
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return message[1]
    return ""


def generate_filled_pdf(current_state: Dict[str, Any]) -> str:
    """Erzeugt das PDF aus dem aktuellen State und gibt den Pfad zur Datei zurück."""
    os.makedirs("out", exist_ok=True)
    unique_id = uuid.uuid4().hex
    json_path = f"out/{unique_id}.json"
    pdf_path = f"out/{unique_id}.pdf"
    save_responses_to_json(state=current_state, output_path=json_path)
    GenericPdfFiller(json_path=json_path).fill(output_path=pdf_path)
    return pdf_path

def set_defaults(state: Dict) -> None:
    """
    Setzt sinnvolle Default-Werte für Slots, falls diese noch NICHT gefüllt sind.
    Überschreibt niemals bereits vorhandene Antworten.
    """
    responses = state.setdefault("responses", {})

    # Beispiel: Geschlecht → "ohne Angabe"
    if not responses.get("sex") or responses["sex"].get("value") in (None, ""):
        responses["sex"] = {
            "value": "ohne Angabe",
            "target_filed_name": [
                "chkGeschlecht1S1",
                "chkGeschlecht2S1",
                "chkGeschlecht3S1",
                "chkGeschlecht4S1",
            ],
            "choices": ["männlich", "weiblich", "divers", "ohne Angabe"],
            # optional: locked steuert, ob der Slot später noch überschreibbar ist
            "locked": False,
        }

def apply_defaults_if_needed() -> None:
    """
    Wendet set_defaults genau EINMAL für das aktuell gewählte Formular an.
    (Wenn form_type wechselt, werden Defaults erneut einmalig gesetzt.)
    """
    s = st.session_state.get("state") or {}
    form_key = s.get("form_type")
    if not form_key:
        return

    if st.session_state.get("_defaults_applied_for") == form_key:
        return  # schon gemacht

    set_defaults(s)
    st.session_state["_defaults_applied_for"] = form_key


# =============================================================================
# Hilfsfunktionen, Nachrichten Ausgabe
# =============================================================================

def stream_assistant_text(full_text: str, delay_seconds: float = 0.01) -> None:
    container = st.chat_message("assistant")
    placeholder = container.empty()
    buffer: List[str] = []
    for character in full_text:
        buffer.append(character)
        placeholder.markdown("".join(buffer))
        time.sleep(delay_seconds)

def stream_new_assistant_messages(prev_len: int, delay_seconds: float = 0.01) -> None:
    for message in st.session_state.history[prev_len:]:
        if role_of(message) == "assistant":
            stream_assistant_text(content_of(message), delay_seconds=delay_seconds)


def run_bot_turn(user_text: str | None = None, *, stream: bool = True, delay: float = 0.01) -> None:
    """Führt chatbot_fn aus und streamt die NEUEN Assistant-Nachrichten dieses Turns."""
    prev_len = len(st.session_state.history)
    new_hist, new_state, _ = chatbot_fn(user_text, st.session_state.history, st.session_state.state)
    st.session_state.history, st.session_state.state = new_hist, new_state
    if stream:
        stream_new_assistant_messages(prev_len, delay_seconds=delay)
    st.rerun()

def emit_and_advance(msg: str, *, delay: float = 0.01) -> None:
    """Erst Abschlussnachricht streamen, dann nächster Slot-Prompt (Bot-Turn)."""
    # 1) Abschluss-/Bestätigungsnachricht
    stream_assistant_text(msg, delay_seconds=delay)
    st.session_state.history.append(("assistant", msg))
    # 2) Nächster Prompt
    run_bot_turn(None, stream=True, delay=delay)  # streamt den Prompt und rerun't

def emit_assistant(
    msg: str,
    *,
    stream: bool = True,
    guard_id: str | None = None,
    delay: float = 0.01,
    stop_after: bool = True,  # <— NEU: nur stoppen, wenn kein UI mehr folgt
) -> None:
    """
    Zeigt eine einzelne Assistant-Nachricht (z.B. Wizard-Zwischenstep).
    - streamt sie (optional),
    - schreibt sie GENAU EINMAL in die History (Guard),
    - stoppt optional den Run, um Doppel-Rendering zu vermeiden.
    """
    key = f"emit:{guard_id or hash((msg, 'assistant'))}"
    if st.session_state.get(key):
        return  # in diesem Run bereits ausgegeben

    if stream:
        stream_assistant_text(msg, delay_seconds=delay)  # erzeugt bereits den assistant-Block
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)

    st.session_state.history.append(("assistant", msg))
    st.session_state[key] = True

    if stop_after:
        st.stop()  # nur stoppen, wenn kein UI mehr direkt gerendert werden soll

# =============================================================================
# Hilfsfunktionen für pdf auslesen
# =============================================================================
def load_file_as_images(uploaded_file, dpi = 150) -> list:

    file_bytes = uploaded_file.getvalue()

    # 1️⃣ Versuch: PDF öffnen und jede Seite als Bild konvertieren
    try:
        import fitz  # PyMuPDF
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")

        images = []
        for page_index, page in enumerate(pdf_doc, start=1):
            pix = page.get_pixmap(alpha=False, matrix = matrix)
            img_bytes = pix.tobytes("png")
            np_arr = np.frombuffer(img_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images.append(cv_img)

        return images

    except Exception:
        # Kein PDF oder Fehler beim Öffnen → normal als Bild lesen
        raise "Error in opening pdf"

# =============================================================================
# Mini-Chat (im Expander) — OpenAI-Client & Antwortlogik
# =============================================================================

if "mini_chat_model" not in st.session_state:
    st.session_state.mini_chat_model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")


def get_openai_client() -> OpenAI:
    """Initialisiert den OpenAI-Client (Key aus st.secrets, optional base_url)."""
    base_url = st.secrets.get("OPENAI_BASE_URL")
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY fehlt in st.secrets")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def mini_chat_respond(
    slot_id: str,
    user_text: str,
    slot_description: str,
    additional_info: List[Dict[str, str]] | None,
    model: str | None = None,
) -> str:
    """
    Führt einen Turn der Mini-Konversation aus und persistiert den Verlauf pro Slot
    in st.session_state.faq_threads[slot_id]. Antwortet nur slotrelevant.
    """
    if "faq_threads" not in st.session_state:
        st.session_state.faq_threads = {}
    if slot_id not in st.session_state.faq_threads:
        st.session_state.faq_threads[slot_id] = []  # Liste aus {role, content}

    # User eingeben
    st.session_state.faq_threads[slot_id].append({"role": "user", "content": user_text})

    # Systemkontext aufbereiten
    extra_text = ""
    for section in (additional_info or []):
        title = section.get("title", "")
        body = section.get("body", "")
        extra_text += f"\n- {title}: {body}"

    system_prompt = (
        "You are a helpful assistant embedded in a government form-filling UI. "
        "Answer ONLY questions that are directly related to the current slot. "
        "If the user's question is unrelated, politely say you don't have that information and suggest focusing on this slot.\n\n"
        f"Slot description:\n{slot_description}\n\n"
        f"Additional info (may include hints):\n{extra_text.strip() or '—'}\n"
    )

    # Nachrichtenliste bilden (System + Verlauf)
    messages = [{"role": "system", "content": system_prompt}]
    for entry in st.session_state.faq_threads[slot_id]:
        messages.append({"role": entry["role"], "content": entry["content"]})

    # Anfrage an OpenAI Responses API
    try:
        client = get_openai_client()
        used_model = model or st.session_state.mini_chat_model
        response = client.responses.create(model=used_model, input=messages)

        # Bevorzugt: output_text; Fallback: aus Items zusammensetzen
        answer_text = getattr(response, "output_text", None)
        if not answer_text:
            answer_text = ""
            for item in getattr(response, "output", []) or []:
                for chunk in getattr(item, "content", []) or []:
                    if getattr(chunk, "type", "") == "output_text" and getattr(chunk, "text", ""):
                        answer_text += chunk.text
            if not answer_text:
                answer_text = "Entschuldigung, ich konnte dazu gerade keine Antwort erzeugen."

        # Assistant im Thread speichern
        st.session_state.faq_threads[slot_id].append({"role": "assistant", "content": answer_text})
        return answer_text

    except Exception as exc:
        error_text = f"Fehler im Mini-Chat: {exc}"
        st.session_state.faq_threads[slot_id].append({"role": "assistant", "content": error_text})
        return error_text
# =============================================================================
# Hilfsfunktionen, Date renderer
# =============================================================================
def _format_date_ddmmyyyy(d: _date | None) -> str:
    if not d:
        return ""
    return d.strftime("%d.%m.%Y")
# =============================================================================
# UI-Renderer: Slot-Widgets, Zusatzinfos-Expander, Mini-Chat
# =============================================================================

def render_slot_interaction_ui() -> None:
    """Rendert das UI für den aktuell angefragten Slot (Widget + Expander)."""
    ui_directive = (st.session_state.state or {}).get("ui")
    if not ui_directive:
        return

    slot_name = ui_directive.get("slot_name") or "slot"
    component_type = ui_directive.get("component")
    component_args = ui_directive.get("args", {})
    additional_info = ui_directive.get("additional_information", [])
    slot_description = ui_directive.get("slot_description", "")

    stable_key_prefix = f"ui_{slot_name}"  # stabil über Reruns

    with st.chat_message("assistant"):
        # 1) Interaktives Slot-Widget
        if component_type == "radio":
            selected_value = st.radio(key=stable_key_prefix + "_radio", **component_args)
            send_clicked = st.button("Auswahl senden", key=stable_key_prefix + "_send")
            if send_clicked and selected_value:
                # new_history, new_state, _ = chatbot_fn(
                #     str(selected_value), st.session_state.history, st.session_state.state
                # )
                # st.session_state.history = new_history
                # st.session_state.state = new_state
                # st.rerun()
                prev_len = len(st.session_state.history)
                new_history, new_state, _ = chatbot_fn(
                    str(selected_value), st.session_state.history, st.session_state.state
                )
                st.session_state.history = new_history
                st.session_state.state = new_state

                # neue Prompts streamen
                stream_new_assistant_messages(prev_len)
                st.rerun()

        elif component_type == "text_input":
            user_text_value = st.text_input(key=stable_key_prefix + "_text", **component_args)
            send_clicked = st.button("Senden", key=stable_key_prefix + "_send")
            if send_clicked and user_text_value:
                # new_history, new_state, _ = chatbot_fn(
                #     user_text_value, st.session_state.history, st.session_state.state
                # )
                # st.session_state.history = new_history
                # st.session_state.state = new_state
                # st.rerun()
                prev_len = len(st.session_state.history)
                new_history, new_state, _ = chatbot_fn(
                    user_text_value, st.session_state.history, st.session_state.state
                )
                st.session_state.history = new_history
                st.session_state.state = new_state

                # neue Prompts streamen
                stream_new_assistant_messages(prev_len)
                st.rerun()
                
        elif component_type == "date_input":
            # date_input liefert datetime.date | None
            picked_date = st.date_input(key=stable_key_prefix + "_date", **component_args)

            send_clicked = st.button("Datum übernehmen", key=stable_key_prefix + "_send")
            if send_clicked:
                # String in TT.MM.JJJJ formen (PDF-kompatibel) – bleibt kompatibel mit deiner Pipeline
                formatted = _format_date_ddmmyyyy(picked_date) if isinstance(picked_date, _date) else ""

                prev_len = len(st.session_state.history)
                new_history, new_state, _ = chatbot_fn(
                    formatted,  # <-- Bot bekommt weiterhin "User-Text", hier unser formatiertes Datum
                    st.session_state.history,
                    st.session_state.state
                )
                st.session_state.history = new_history
                st.session_state.state = new_state

                # Prompt/Nachrichten streamen wie gehabt
                stream_new_assistant_messages(prev_len)
                st.rerun()
        
        elif component_type == "number_input":
            picked_number = st.number_input("Zahl eingeben",key=stable_key_prefix + "_number", **component_args)
            send_clicked = st.button("Übernehmen", key=stable_key_prefix + "_send")
            picked_number = str(picked_number)
            if send_clicked:
                prev_len = len(st.session_state.history)
                new_history, new_state, _ = chatbot_fn(
                    picked_number,  # <-- Bot bekommt weiterhin "User-Text", hier unser formatiertes Datum
                    st.session_state.history,
                    st.session_state.state
                )
                st.session_state.history = new_history
                st.session_state.state = new_state

                # Prompt/Nachrichten streamen wie gehabt
                stream_new_assistant_messages(prev_len)
                st.rerun()

        # 2) Expander: Zusatzinfos (oben) + Mini-Chat (unten)
        if additional_info:
            with st.expander("Mehr Infos & Rückfragen", expanded=False):
                # Zusatzinformationen ausgeben
                for idx, section in enumerate(additional_info):
                    title = section.get("title", f"Info {idx + 1}")
                    body = section.get("body", "")
                    st.markdown(f"**{title}**")
                    st.markdown(body, unsafe_allow_html=True)
                    if idx < len(additional_info) - 1:
                        st.divider()

                st.divider()

                # Mini-Chat NUR anzeigen, wenn es Zusatzinfos gibt
                if "faq_threads" not in st.session_state:
                    st.session_state.faq_threads = {}
                thread_id = slot_name
                thread = st.session_state.faq_threads.get(thread_id, [])

                if thread:
                    for message in thread:
                        speaker = "Du" if message["role"] == "user" else "Fachinfo-Bot"
                        st.markdown(f"**{speaker}:** {message['content']}")
                    st.markdown("---")

                question_key = stable_key_prefix + "_faq_q"
                send_button_key = stable_key_prefix + "_faq_send"
                question_text = st.text_input("Rückfrage stellen …", key=question_key)

                if st.button("Frage senden", key=send_button_key):
                    if question_text and question_text.strip():
                        _ = mini_chat_respond(
                            slot_id=thread_id,
                            user_text=question_text.strip(),
                            slot_description=slot_description or component_args.get("label", ""),
                            additional_info=additional_info,
                            model=st.session_state.mini_chat_model,
                        )
                        st.rerun()
                    else:
                        st.info("Bitte eine Frage eingeben.")

# =============================================================================
# UI-Renderer: Abschluss-Flow (PDF-Download + Upload-Mockup)
# =============================================================================

def render_completion_ui() -> None:
    """Zeigt nach vollständiger Dateneingabe Download + Upload-Mock an."""
    form_state = st.session_state.state or {}
    if not form_state.get("completed"):
        return

    language_code = form_state.get("lang") or "de"

    # PDF nur einmal erzeugen und cachen
    if "generated_pdf_path" not in st.session_state:
        st.session_state.generated_pdf_path = generate_filled_pdf(form_state)

    pdf_path = st.session_state.generated_pdf_path

    with st.container(border=True):
        st.subheader("Formular fertiggestellt")
        try:
            with open(pdf_path, "rb") as pdf_fp:
                pdf_bytes = pdf_fp.read()
            st.download_button(
                label=download_button_msgs.get(language_code, download_button_msgs["de"]),
                data=pdf_bytes,
                file_name="formular.pdf",
                mime="application/pdf",
                key="pdf_download_button",
            )
            st.caption(pdf_file_msgs.get(language_code, pdf_file_msgs["de"]))
        except Exception as exc:
            st.error(f"PDF konnte nicht geladen werden: {exc}")

        st.divider()

        # Upload-Mockup, falls angefordert
        show_upload_mock = bool(form_state.get("show_upload"))
        if show_upload_mock:
            st.write("**Upload (Mock)**")
            upload_label = form_state.get("upload_label") or "Dateien hochladen"
            files_label = files_msgs.get(language_code, files_msgs["de"]) 

            uploaded_files = st.file_uploader(
                label=upload_label,
                accept_multiple_files=True,
                key="final_upload_files",
            )
            if uploaded_files:
                st.session_state.state["uploaded_files"] = uploaded_files
                st.write(f"**{files_label}:**")
                for file_obj in uploaded_files:
                    st.write(f"- {getattr(file_obj, 'name', 'Datei')}")

            if st.button("Upload abschließen", key="finalize_upload_button"):
                # Flags analog Gradio-Flow
                st.session_state.state["awaiting_final_upload"] = False
                st.session_state.state["show_upload"] = False
                st.session_state.state["completed"] = False  # Download-Bereich ausblenden

                # Abschlussnachricht in den Chat
                final_message = final_msgs.get(language_code, final_msgs["de"]) 
                # st.session_state.history.append(("assistant", final_message))
                # st.rerun()
                # assistant_msg_then_next_slot(final_message)
                emit_and_advance(final_message)

# =============================================================================
# UI-Renderer: ShortCutWizard (capture/ upload image/ query creditreform api/ continue)
# =============================================================================

def render_shortcut_wizard_ui() -> None:
    """
    UI für den ShortCutWizard:
    - 4 Buttons (Foto, Upload, CR-Mock, Manuell)
    - Kamera/Upload
    - Data-Editor (bearbeitbar)
    - Nachfrage Betriebsstättenadresse
    - Bestätigen → Mapping-Hook (TODO: du füllst Slots)
    """
    sstate = st.session_state.state or {}
    if sstate.get("active_wizard") != "shortcut_wizard":
        return
    
    apply_defaults_if_needed()

    handles = sstate.get("wizard_handles") or {}
    wiz: ShortCutWizard = handles.get("shortcut_wizard")
    if not wiz:
        wiz = ShortCutWizard(ShortCutWizardState(lang_code=sstate.get("lang") or "de"))
        handles["shortcut_wizard"] = wiz
        st.session_state.state["wizard_handles"] = handles

    # 1) Schritttext ausgeben (Assistant)
    msg, done, _ = wiz.step(None)
    # with st.chat_message("assistant"):
    #     st.markdown(msg)
    emit_assistant(msg, stream=True, guard_id=f"shortcut:{wiz.state.phase}", stop_after=False)

    # 2) Phasen-spezifisches UI
    phase = wiz.state.phase

    # a) Entscheidung: 4 Buttons
    if phase == "ask_path":
        with st.chat_message("assistant"):
            c1, c2, c3, c4, c5 = st.columns(5)
            if c1.button("📷 Foto aufnehmen", key="scw_btn_camera"):
                wiz.state.choice = "camera"
                wiz.state.phase = "capture"
                st.rerun()

            if c2.button("🖼️ Bild hochladen", key="scw_btn_upload"):
                wiz.state.choice = "upload"
                wiz.state.phase = "upload"
                st.rerun()

            if c3.button("📃 PDF hochladen", key="scw_btn_pdf_upload"):
                wiz.state.choice = "pdf_upload"
                wiz.state.phase = "pdf_upload"
                st.rerun()

            if c4.button("🏦 Creditreform (Mock)", key="scw_btn_crf"):
                wiz.state.choice = "crf"
                wiz.state.phase = "cr_mock"
                # Mock-Daten vorbereiten (minimal, du kannst das anpassen)
                wiz.state.extracted = {
                    "authority": "",
                    "hra_number": "",
                    "company_name": "",
                    "legal_type": "",
                    "address": "",
                    "activity": "",
                    "ceo": [],
                }
                wiz.state.phase = "review"
                st.rerun()

            if c5.button("✍️ Manuell weiter", key="scw_btn_manual"):
                wiz.state.choice = "manual"
                wiz.state.phase = "done"

                form_key = st.session_state.state.get("form_type")
                slots_def = FORMS.get(form_key, {}).get("slots", [])

                wiz.apply_mapping_and_finish(st.session_state.state, slots_def)

                # NEU: immer erst Bestätigungs-Text, dann nächster Slot-Prompt
                # assistant_msg_then_next_slot("Wir machen manuell weiter. ✅")
                emit_and_advance("Wir machen manuell weiter. ✅")


    # b) Foto aufnehmen
    if phase == "capture":
        with st.chat_message("assistant"):
            img_file_buffer = st.camera_input("Bitte fotografieren Sie den Handelsregisterauszug.")
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                # OCR + LLM-Extraktion
                with st.spinner("Informationen werden aus dem Bild extrahiert …"):
                    data = extract_information_HRA_info_from_img(image)
                wiz.state.extracted = data or {}
                wiz.state.phase = "review"
                st.rerun()

    # c) Bild hochladen
    if phase == "upload":
        with st.chat_message("assistant"):
            up = st.file_uploader(
                "Bitte wählen Sie Bilder (PNG/JPG).",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="scw_uploader",
            )
            if up:  # nur weiter, wenn mind. 1 Datei gewählt wurde
                images = []
                for file in up:
                    bytes_data = file.getvalue()
                    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)

                with st.spinner("Informationen werden aus dem Bild extrahiert …"):
                    data = extract_information_HRA_info_from_img(images)

                wiz.state.extracted = data or {}
                wiz.state.phase = "review"
                st.rerun()

    # c) Pdf hochladen
    if phase == "pdf_upload":
        with st.chat_message('assistant'):
            up = st.file_uploader("Bitte wählen Sie eine PDF Datei.", type=["pdf"], accept_multiple_files=False, key="scw_uploader")
            if up is not None:
                images = load_file_as_images(up)
                with st.spinner("Informationen werden aus dem PDF extrahiert …"):
                    data = extract_information_HRA_info_from_img(images)
                wiz.state.extracted = data or {}
                wiz.state.phase = "review"
                st.rerun()

    # d) Review: Data Editor
    if phase == "review":

        wiz.state.extracted["num_representatives"] = len(wiz.state.extracted["ceo"])

        df_to_dict_column_names = {
            "authority":"Registergericht",
            "hra_number":"Registernummer",
            "company_name":"Name des Unternehmens",
            "legal_type":"Rechtsform",
            "address":"Adresse",
            "activity":"Tätigkeit",
            "ceo":"Geschäftsleitung",
            "num_representatives":"Anzahl Geschäftsführer"
        }

        dict_column_names_to_df_names = {v:k for k,v in df_to_dict_column_names.items()}

        with st.chat_message("assistant"):
            st.markdown("### Erkannte Daten")
            flat = (wiz.state.extracted or {}).copy()

            # CEO-Liste ⇄ editierbarer String (roundtrip-fähig, inkl. city)
            # Anzeigeformat: "Nachname, Vorname, Stadt, TT.MM.JJJJ | Nachname, Vorname, Stadt, TT.MM.JJJJ"
            if isinstance(flat.get("ceo"), list):
                def _row_to_str(row: dict) -> str:
                    return ", ".join([
                        str(row.get("family_name", "")).strip(),
                        str(row.get("given_name", "")).strip(),
                        str(row.get("city", "")).strip(),
                        str(row.get("birth_date", "")).strip(),
                    ])
                flat["ceo"] = " | ".join([_row_to_str(r) for r in flat["ceo"]])

            df = pd.DataFrame([flat])
            df.rename(df_to_dict_column_names,axis=1,inplace=True)
            edited_df = st.data_editor(df, num_rows="fixed", key="scw_editor")
            edited_df.rename(dict_column_names_to_df_names, axis=1, inplace=True)
            edited = edited_df.to_dict(orient="records")[0]

            # CEO: zurückwandeln zu Liste[Dict] im Schema der Name-Klasse
            ceo_raw = edited.get("ceo")
            if isinstance(ceo_raw, str):
                ceo_list_dicts = []
                # Split auf Einträge
                entries = [p.strip() for p in ceo_raw.split("|") if p.strip()]
                for entry in entries:
                    # Erwartetes Format: "Nachname, Vorname, Stadt, TT.MM.JJJJ"
                    parts = [x.strip() for x in entry.split(",")]
                    # robust befüllen (weniger Teile → leere Strings)
                    family_name = parts[0] if len(parts) > 0 else ""
                    given_name  = parts[1] if len(parts) > 1 else ""
                    city        = parts[2] if len(parts) > 2 else ""
                    birth_date  = parts[3] if len(parts) > 3 else ""

                    # minimale Normalisierung / Sanity-Check fürs Datum (optional)
                    # akzeptiere bereits korrektes "TT.MM.JJJJ" oder leer
                    if birth_date and not re.match(r"^\d{2}\.\d{2}\.\d{4}$", birth_date):
                        # kleine Heuristik: 2025-01-31 → 31.01.2025
                        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", birth_date)
                        if m:
                            birth_date = f"{m.group(3)}.{m.group(2)}.{m.group(1)}"

                    ceo_list_dicts.append({
                        "family_name": family_name,
                        "given_name":  given_name,
                        "city":        city,
                        "birth_date":  birth_date,
                    })

                edited["ceo"] = ceo_list_dicts

            # (Kompatibilität) Falls es noch eine alte Liste[List[str]] gibt, in Dicts heben
            elif isinstance(ceo_raw, list) and ceo_raw and isinstance(ceo_raw[0], list):
                ceo_list_dicts = []
                for row in ceo_raw:
                    family_name = row[0] if len(row) > 0 else ""
                    given_name  = row[1] if len(row) > 1 else ""
                    city        = row[2] if len(row) > 2 else ""
                    birth_date  = row[3] if len(row) > 3 else ""
                    ceo_list_dicts.append({
                        "family_name": family_name,
                        "given_name":  given_name,
                        "city":        city,
                        "birth_date":  birth_date,
                    })
                edited["ceo"] = ceo_list_dicts

            colL, colR = st.columns(2)
            if colL.button("✅ Daten übernehmen", key="scw_take_over"):
                wiz.state.edited = edited
                wiz.state.phase = "ask_branch_addr"
                st.rerun()
            if colR.button("🔁 Neue Datei verwenden", key="scw_retry_image"):
                choice = wiz.state.choice
                wiz.state.extracted = {}
                wiz.state.edited = {}
                if choice == "camera":
                    wiz.state.phase = "capture"
                elif choice == "upload":
                    wiz.state.phase = "upload"
                elif choice == "pdf_upload":
                    wiz.state.phase = "pdf_upload"
                st.rerun()

    # e) Nachfrage: Betriebsstätten-Adresse?
    if phase == "ask_branch_addr":
        with st.chat_message("assistant"):
            st.markdown("Bezieht sich die angegebene **Adresse** auch auf die **Betriebsstätte**?")
            col1, col2 = st.columns(2)

            if col1.button("Ja", key="scw_addr_yes"):
                wiz.state.edited["_is_branch_addr_same"] = True

                form_key = st.session_state.state.get("form_type")
                slots_def = FORMS.get(form_key, {}).get("slots", [])
                wiz.apply_mapping_and_finish(st.session_state.state, slots_def)

                # NEU: Einheitlich zuerst Wizard/Bestätigungs-Text, dann nächster Slot
                # assistant_msg_then_next_slot("Daten übernommen. Wir machen mit den restlichen Angaben weiter. ✅")
                emit_and_advance("Daten übernommen. Wir machen mit den restlichen Angaben weiter. ✅")

            if col2.button("Nein", key="scw_addr_no"):
                wiz.state.edited["_is_branch_addr_same"] = False

                form_key = st.session_state.state.get("form_type")
                slots_def = FORMS.get(form_key, {}).get("slots", [])
                wiz.apply_mapping_and_finish(st.session_state.state, slots_def)

                # assistant_msg_then_next_slot("Alles klar – die Betriebsstättenanschrift erfassen wir separat. ✅")
                emit_and_advance("Alles klar – die Betriebsstättenanschrift erfassen wir separat. ✅")

# =============================================================================
# IDCardWizard
# =============================================================================

def render_idcard_wizard_ui() -> None:
    """
    UI für den IDCardWizard:
    - 2 Buttons (Upload, Manuell)
    - Upload
    - Data-Editor (bearbeitbar)
    - Bestätigen → Mapping-Hook (TODO: du füllst Slots)
    """
    sstate = st.session_state.state or {}
    if sstate.get("active_wizard") != "idcard_wizard":
        return

    handles = sstate.get("wizard_handles") or {}
    wiz: IDCardWizard = handles.get("idcard_wizard")
    if not wiz:
        wiz = IDCardWizard(IDCardWizardState(lang_code=sstate.get("lang") or "de"))
        handles["idcard_wizard"] = wiz
        st.session_state.state["wizard_handles"] = handles

    # 1) Schritttext ausgeben (Assistant)
    msg, done, _ = wiz.step(None)
    # with st.chat_message("assistant"):
    #     st.markdown(msg)
    emit_assistant(msg, stream=True, guard_id=f"idcard:{wiz.state.phase}", stop_after=False)

    # 2) Phasen-spezifisches UI
    phase = wiz.state.phase

    # a) Entscheidung: 4 Buttons
    if phase == "ask_path":
        with st.chat_message("assistant"):
            c1, c2 = st.columns(2)

            if c1.button("🪪 Ausweisbilder hochladen", key="idw_btn_upload"):
                wiz.state.choice = "upload"
                wiz.state.phase = "upload"
                st.rerun()

            if c2.button("✍️ Manuell weiter", key="idw_btn_manual"):
                wiz.state.choice = "manual"
                wiz.state.phase = "done"

                form_key = st.session_state.state.get("form_type")
                slots_def = FORMS.get(form_key, {}).get("slots", [])

                wiz.apply_mapping_and_finish(st.session_state.state, slots_def)

                # NEU: immer erst Bestätigungs-Text, dann nächster Slot-Prompt
                # assistant_msg_then_next_slot("Wir machen manuell weiter. ✅")
                emit_and_advance("Wir machen manuell weiter. ✅")

    # b) Bild hochladen
    if phase == "upload":
        with st.chat_message("assistant"):
            up = st.file_uploader("Bitte wählen Sie die Bilder (PNG/JPG).", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="idw_uploader")
            if up:
                images = []
                for idx, file in enumerate(up):
                    img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ### debug ###
                    cv2.imwrite(f"uploaded_{idx}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    #############
                    images.append(img)
                with st.spinner("Informationen werden extrahiert …"):
                    data = extract_information_id_card(images)
                print(data)
                ### debug ###
                wiz.state.extracted = data or {}
                wiz.state.phase = "review"
                st.rerun()

    # c) Review: Data Editor
    if phase == "review":

        df_to_dict_column_names = {
            "given_name":"Vorname",
            "family_name":"Nachname",
            "birth_date":"Geburtsdatum",
            "nationality":"Staatsangehörigkeit",
            "address":"Adresse",
            "birth_place":"Geburtsort"
        }

        dict_column_names_to_df_names = {v:k for k,v in df_to_dict_column_names.items()}

        with st.chat_message("assistant"):
            st.markdown("### Erkannte Daten")
            flat = (wiz.state.extracted or {}).copy()

            # addresse auslesen
            def _to_str(row: dict) -> str:
                return ", ".join([
                    str(row.get("street_name", "")).strip(),
                    str(row.get("street_number", "")).strip(),
                    str(row.get("postalcode", "")).strip(),
                    str(row.get("city","")).strip()
                ])
            # adresse auslesen
            address = flat.get("address")
            flat["address"] = _to_str(address)

            # nationality auslesen
            flat['nationality'] = 'Deutsch' if flat['germany'] else flat['nationality']


            flat['birth_place'] = f"{flat['birth_place']}, DEUTSCHLAND" if flat["germany"] else f"{flat['birth_place']}, {flat['nationality']}"
            # pop filed germany
            del flat["germany"]



            df = pd.DataFrame([flat])
            df.rename(df_to_dict_column_names,axis=1,inplace=True)
            edited_df = st.data_editor(df, num_rows="fixed", key="scw_editor")
            edited_df.rename(dict_column_names_to_df_names, axis=1, inplace=True)
            edited = edited_df.to_dict(orient="records")[0]

            colL, colR = st.columns(2)
            if colL.button("✅ Daten übernehmen", key="idw_take_over"):
                # an slots anpassen
                if edited.get('nationality').strip().lower() in ['deutsch','deutschland','bundesrepublik deutschland']:
                    edited['nationality'] = True
                else:
                    edited["other_nationality"] = edited['nationality']
                    edited["nationality"] = False

                wiz.state.edited = edited
                form_key = st.session_state.state.get("form_type")
                slots_def = FORMS.get(form_key, {}).get("slots", [])
                wiz.apply_mapping_and_finish(st.session_state.state, slots_def)
                emit_and_advance("Daten übernommen. Wir machen mit den restlichen Angaben weiter. ✅")

            if colR.button("🔁 Neue Bilder hochladen", key="idw_retry_image"):
                wiz.state.extracted = {}
                wiz.state.edited = {}
                wiz.state.phase = "upload"
                st.rerun()

# =============================================================================
# Hauptablauf
# =============================================================================

def main() -> None:
    # ---------- Session-Variablen einheitlich initialisieren ----------
    if "state" not in st.session_state:
        st.session_state.state =  {
            "form_type": None,   # which form the user has chosen
            "lang": None,        # language code for future multi-language support
            "responses": {},     # stores slot_name -> user_response
            "idx": 0,            # pointer into the slots list
            "pdf_file": None,     # path to the pdf file for later overwrite
            "active_wizard":None, # whether there is currently a wizard active 
            "wizard_handles":None # handles object of the current wizard
        }  # immer Dict, niemals None
    if "history" not in st.session_state:
        st.session_state.history = []

    # apply_defaults_if_needed()

    # ---------- Start-Turn: Begrüßung/Sprachauswahl vom Bot ----------
    if "app_started" not in st.session_state:
        # prev_len = len(st.session_state.history)
        # st.session_state.app_started = True

        # # 1) Begrüßung VOR ALLEM als erste Assistenten-Nachricht einfügen
        # # st.session_state.history.append(("assistant", GREETING_TEXT))

        # # 2) Danach den normalen Bot-Start-Turn triggern (Sprachauswahl etc.)
        # new_history, new_state, _ = chatbot_fn(
        #     None, st.session_state.history, st.session_state.state
        # )
        # st.session_state.history = new_history
        # st.session_state.state = new_state
        # stream_new_assistant_messages(prev_len)
        st.session_state.app_started = True
        run_bot_turn(None, stream=True)  # streamt Begrüßung/Sprachauswahl und macht st.rerun()
    # ---------- Bisherigen Verlauf rendern ----------
    for message in st.session_state.history:
        with st.chat_message(role_of(message) or "assistant"):
            st.markdown(content_of(message))

    # ---------- Nutzereingabe ----------
    user_input_text = st.chat_input(CHAT_INPUT_PLACEHOLDER)
    if user_input_text:
        # Benutzer sofort anzeigen
        st.session_state.history.append(("user", user_input_text))
        with st.chat_message("user"):
            st.markdown(user_input_text)

        # Bot ausführen (mit Spinner)
        with st.spinner("Bitte einen Moment …"):
            previous_length = len(st.session_state.history)
            new_history, new_state, _ = chatbot_fn(
                user_input_text, st.session_state.history, st.session_state.state
            )
            st.session_state.history = new_history
            st.session_state.state = new_state

        # Neue Assistant-Nachrichten streamen
        for message in st.session_state.history[previous_length:]:
            if role_of(message) == "assistant":
                stream_assistant_text(content_of(message))

    # ---------- Kontextbezogene UI-Elemente ----------
    render_slot_interaction_ui()     # Widget + Zusatzinfos + Mini-Chat
    render_completion_ui()           # PDF-Download + Upload-Mockup (bei completed)
    render_shortcut_wizard_ui()
    render_idcard_wizard_ui()
    # ---------- Print alle bisher erfassten Felder -----
    # ---------- Debugging Print ----------------------
    render_debug_panel()

    # ----------- Falls der Wizard eben fertig wurde, aber noch keine Frage gestellt ist
    if (st.session_state.state or {}).get("awaiting_first_slot_prompt") and \
    not (st.session_state.state or {}).get("active_wizard"):
        # prev_len = len(st.session_state.history)
        # new_history, new_state, _ = chatbot_fn(
        #     None, st.session_state.history, st.session_state.state
        # )
        # st.session_state.history = new_history
        # st.session_state.state = new_state

        # stream_new_assistant_messages(prev_len)
        # st.rerun()
        run_bot_turn(None, stream=True)  


if __name__ == "__main__":
    main()
