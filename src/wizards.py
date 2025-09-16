"""wizzards are mainly short, stateful multiturn conversations with an llm.
If a user started a wizzard in a former chat bot function call, the wizzard is marked as active in the chatbot functions state and the information
needed to perform the wizzards task is stored in a wizzard handles object"""

# language_wizard.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import json
from openai import OpenAI
import json, re

def code_to_label(code: str) -> str:
    return {"de":"Deutsch","en":"English","fr":"Français","tr":"Türkçe"}.get(code, code)

@dataclass
class LanguageWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    awaiting_confirmation: bool = False
    # Conversation-Subloop-State (Responses API, Multi-Turn)
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    history: list = field(default_factory=list)  # kleine Verlaufsliste nur für den Subloop

class LanguageWizard:
    def __init__(self, state: Optional[LanguageWizardState] = None, model: str = "gpt-4o-mini"):
        self.state = state or LanguageWizardState()
        self.model = model
        self.client = OpenAI()

    # --- LLM-Aufrufe ---------------------------------------------------------
    def _llm_detect_language(self, user_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Nutzt Responses-API mit JSON-Structured-Output.
        Gibt (lang_code, lang_label, confirm_prompt, response_id) zurück.
        """
        system_prompt = (
            "You are a language detector and copywriter. "
            "1) Decide in which language the USER WANTS TO COMMUNICATE. Prefer explicit user intent over text language. "
            "2) Return a short confirmation question in that language asking the user if they want to continue in that language. "
            "3) Use ISO 639-1 code (de/en/fr/tr if applicable)."
        )
        schema = {
            "type":"object",
            "additionalProperties": False,
            "properties":{
                "language_code":{"type":"string"},
                "language_label":{"type":"string"},
                "confirmation_prompt":{"type":"string"}
            },
            "required":["language_code","language_label","confirmation_prompt"]
        }

        print(self.state.conversation_id, self.state.previous_response_id)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_text}
            ],
            store=True,  # Conversation serverseitig speichern (vereinfacht Multi-Turn)
            text={
                "format":{
                    "type":"json_schema",
                    "name":"structured_output",
                    "schema":schema,
                    "strict":True
                }
            },
            # Conversation-state verbinden, falls vorhanden:
            previous_response_id=self.state.previous_response_id 
        )

        # Responses-API liefert bei JSON-Mode den Text als JSON-String:
        data = json.loads(resp.output_text)
        lang_code = data.get("language_code")
        lang_label = data.get("language_label") or code_to_label(lang_code or "")
        confirm_prompt = data.get("confirmation_prompt")
        response_id = getattr(resp, "id", None)

        # Conversation-ID (falls neu erstellt) merken:
        conv = getattr(resp, "conversation", None)
        if not self.state.conversation_id and conv and getattr(conv, "id", None):
            self.state.conversation_id = conv.id

        return lang_code, lang_label, confirm_prompt, response_id

    def _llm_check_approval(self, user_text: str) -> Optional[bool]:
        """LLM-Backup für Ja/Nein-Klassifikation in beliebiger Sprache."""
        schema = {
            "type":"object",
            "additionalProperties": False,
            "properties":{
                "approved":{"type":"boolean"},
                "confirmation_prompt":{"type":"string"}},
            "required":["approved","confirmation_prompt"]
        }
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role":"system","content":"Classify if the user message is a YES/approval or NO/rejection. If approval, formulate a short message like 'Great — we'll continue in English.' in the users language (confirmation_prompt)."},
                {"role":"user","content":user_text}
            ],
            store=True,
            text={"format":{"type":"json_schema","name":"yn","schema":schema,"strict":True}},
            previous_response_id=self.state.previous_response_id
        )
        out = json.loads(resp.output_text)
        return bool(out.get("approved")), out.get("confirmation_prompt")
    
    def _normalize(self, text: str) -> str:
        return (text or "").strip().lower()

    def _fast_language_from_text(self, user_text: str) -> Optional[str]:
        t = self._normalize(user_text)

        # reichlich Varianten / Sprachen
        de_keys = {
            "de", "deutsch", "auf deutsch", "sprich deutsch", "german", "in german",
            "bitte deutsch", "deutsche sprache"
        }
        en_keys = {
            "en", "englisch", "english", "speak english", "in english",
            "please english", "anglo", "eng", "eng language"
        }
        fr_keys = {
            "fr", "französisch", "franzoesisch", "français", "francais",
            "en français", "in french", "french", "parlons français"
        }
        tr_keys = {
            "tr", "türkçe", "turkce", "turkish", "ingilizce degil türkçe", "türk dili",
            "türkisch", "auf türkisch", "in turkish"
        }

        # substring-checks (robust gegen Satzformen)
        def has_any(keys): return any(k in t for k in keys)

        if has_any(de_keys): return "de"
        if has_any(en_keys): return "en"
        if has_any(fr_keys): return "fr"
        if has_any(tr_keys): return "tr"
        return None

    def _build_confirm_prompt(self, code: str) -> str:
        return {
            "de": "Ich habe **Deutsch** erkannt. Sollen wir auf Deutsch weitermachen? (Ja/Nein)",
            "en": "I detected **English**. Shall we continue in English? (Yes/No)",
            "fr": "J’ai détecté le **français**. Souhaitez-vous continuer en français ? (Oui/Non)",
            "tr": "**Türkçe** algıladım. Türkçe devam edelim mi? (Evet/Hayır)",
        }.get(code, "Language detected. Continue? (Yes/No)")

    def _fast_approval(self, user_text: str) -> Optional[bool]:
        t = self._normalize(user_text)

        # breite Ja-/Nein-Mengen (mehrsprachig, inkl. Umgangssprache)
        YES = {
            "ja","j","jawohl","jo","jup","jep","klar","korrekt","richtig","okay","ok","okey",
            "yes","y","yeah","yep","sure","correct","right","affirmative",
            "oui","ouais","d'accord","dac","bien sûr",
            "evet","tamam","olur","aynen"
        }
        NO = {
            "nein","n","nee","nö","nicht","falsch","auf keinen fall",
            "no","nope","nah","never",
            "non","pas","pas du tout",
            "hayır","hayir","yok","olmaz","asla"
        }

        # exakte matches
        if t in YES: return True
        if t in NO:  return False

        # häufige Satzanfänge
        yes_sub = ["ja,", "ja.", "ja!", "yes,", "yes.", "oui,", "evet,", "ok,", "okay,"]
        no_sub  = ["nein,", "nein.", "no,", "no.", "non,", "hayır,", "hayir,", "yok,", "olmaz,"]

        if any(t.startswith(s) for s in yes_sub): return True
        if any(t.startswith(s) for s in no_sub):  return False

        return None

    # --- Wizard-Schritte -----------------------------------------------------
    def step(self, user_text: Optional[str]) -> Tuple[str, bool]:
        s = self.state

        # 2) Sprache erkennen (Fast-Path -> LLM-Fallback)
        if not s.lang_code and user_text:
            s.history.append(("user", user_text))

            # 🔹 FAST: heuristische Spracherkennung
            fast_code = self._fast_language_from_text(user_text)
            if fast_code:
                s.lang_code = fast_code
                s.awaiting_confirmation = True
                confirm = self._build_confirm_prompt(fast_code)
                s.history.append(("assistant", confirm))
                return confirm, False, s.lang_code

            # 🔹 FALLBACK: LLM
            code, label, confirm, rid = self._llm_detect_language(user_text)
            s.previous_response_id = rid or s.previous_response_id

            if code and confirm:
                s.lang_code = code
                s.awaiting_confirmation = True
                s.history.append(("assistant", confirm))
                return confirm, False, s.lang_code
            else:
                msg = "Ich bin mir unsicher. Bitte nenne die gewünschte Sprache (z. B. Deutsch, English, Français, Türkçe)."
                s.history.append(("assistant", msg))
                return msg, False, s.lang_code

        # 3) Approval prüfen (Fast-Path -> LLM-Fallback)
        if s.awaiting_confirmation and user_text:
            # 🔹 FAST: Ja/Nein-Heuristik
            fast_yn = self._fast_approval(user_text)
            if fast_yn is True:
                s.awaiting_confirmation = False
                done_msg = {
                    "de":"Alles klar – wir sprechen Deutsch. ✅",
                    "en":"Great — we'll continue in English. ✅",
                    "fr":"Parfait — nous continuons en français. ✅",
                    "tr":"Harika — Türkçe devam edelim. ✅"
                }.get(s.lang_code, "Okay — language set. ✅")
                s.history.append(("assistant", done_msg))
                return done_msg, True, s.lang_code

            if fast_yn is False:
                s.lang_code = None
                s.awaiting_confirmation = False
                msg = "Kein Problem. Welche Sprache hättest du gern?"
                s.history.append(("assistant", msg))
                return msg, False, s.lang_code

            # 🔹 FALLBACK: LLM
            approved, done_msg = self._llm_check_approval(user_text)
            if approved is True:
                s.awaiting_confirmation = False
                s.history.append(("assistant", done_msg))
                return done_msg, True, s.lang_code

            if approved is False:
                s.lang_code = None
                s.awaiting_confirmation = False
                msg = "Kein Problem. Welche Sprache hättest du gern?"
                s.history.append(("assistant", msg))
                return msg, False, s.lang_code

            # Unklar
            msg = "Bitte antworte mit Ja/Nein."
            s.history.append(("assistant", msg))
            return msg, False, s.lang_code

        # Fallback
        msg = "Wie sollen wir sprechen?"
        s.history.append(("assistant", msg))
        return msg, False, s.lang_code

    def export_state(self) -> Dict[str, Any]:
        return {
            "turns": self.state.turns,
            "lang_code": self.state.lang_code,
            "awaiting_confirmation": self.state.awaiting_confirmation,
            "conversation_id": self.state.conversation_id,
            "previous_response_id": self.state.previous_response_id,
            "history": self.state.history[-8:],  # klein halten
        }

@dataclass
class FormSelectionWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    available_form_keys: List[str] = field(default_factory=list)
    translated_labels: List[str] = field(default_factory=list)  # gleiche Reihenfolge wie keys
    awaiting_selection: bool = False
    selected_form_key: Optional[str] = None
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None

class FormSelectionWizard:
    """
    Fragt das gewünschte Formular ab.
    Kommuniziert in der im Bot-State hinterlegten Sprache (lang_code).
    Nutzt das LLM, um die Formularnamen dynamisch zu übersetzen + einen lokalen Prompt zu erzeugen.
    """
    def __init__(self, state: Optional[FormSelectionWizardState] = None, model: str = "gpt-4o-mini"):
        self.state = state or FormSelectionWizardState()
        self.model = model
        self.client = OpenAI()

    def _llm_localize_form_list(self, lang_code: str, form_keys: List[str]) -> Tuple[List[str], str, Optional[str]]:
        """
        Lässt das LLM (Responses API) die sichtbaren Formularnamen in die Zielsprache übersetzen
        und einen kurzen, lokalisierten Prompt generieren.
        Gibt (labels, prompt, response_id) zurück.
        """
        system_prompt = (
            "You localize UI labels. Task:\n"
            "1) Translate the given form titles for end-users into the target language (ISO 639-1 code provided).\n"
            "2) Return a concise question asking which form to fill in the same language.\n"
            "Keep order; keep it user-friendly; no extra commentary."
        )
        schema = {
            "type":"object",
            "additionalProperties": False,
            "properties":{
                "labels":{"type":"array","items":{"type":"string"}},
                "prompt":{"type":"string"}
            },
            "required":["labels","prompt"]
        }
        user_payload = {
            "target_lang": lang_code,
            "form_titles": form_keys  # hier reichen die Schlüssel, wenn sie schon „sprechende“ Namen sind;
                                      # sonst könntest du Display-Titel übergeben.
        }
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}
            ],
            store=True,
            text={"format":{"type":"json_schema","name":"localized","schema":schema,"strict":True}},
            previous_response_id=self.state.previous_response_id
        )
        data = json.loads(resp.output_text)
        labels = data.get("labels", [])
        prompt = data.get("prompt", "Bitte wählen Sie ein Formular:")
        rid = getattr(resp, "id", None)

        # conversation_id setzen, falls neu
        conv = getattr(resp, "conversation", None)
        if not self.state.conversation_id and conv and getattr(conv, "id", None):
            self.state.conversation_id = conv.id

        return labels, prompt, rid

    def _format_numbered_list(self, items: List[str]) -> str:
        return "\n".join(f"{i+1}. {it}" for i, it in enumerate(items))

    def step(self, user_text: Optional[str]) -> Tuple[str, bool, Optional[str]]:
        s = self.state

        # 1) Erster Turn: Liste erzeugen und anzeigen
        if s.turns == 0:
            s.turns += 1

            # ✅ NEU: schneller Fallback für Deutsch – KEIN LLM nötig
            if (s.lang_code or "de").lower() == "de":
                s.translated_labels = list(s.available_form_keys)  # 1:1 übernehmen
                s.awaiting_selection = True
                prompt = "Bitte wählen Sie ein Formular:"
                numbered = self._format_numbered_list(s.translated_labels)
                hint = "Sie können die Nummer oder den Namen eingeben."
                return f"{prompt}\n{numbered}\n\n{hint}", False, s.lang_code

            # 🔁 Andernfalls wie gehabt: LLM-Übersetzung
            labels, prompt, rid = self._llm_localize_form_list(
                s.lang_code or "de",
                s.available_form_keys
            )
            s.translated_labels = labels
            s.previous_response_id = rid or s.previous_response_id
            s.awaiting_selection = True

            numbered = self._format_numbered_list(labels)
            hint = {
                "de": "Sie können die Nummer oder den Namen eingeben.",
                "en": "You can enter the number or the name.",
                "fr": "Vous pouvez saisir le numéro ou le nom.",
                "tr": "Numarayı veya adı girebilirsiniz."
            }.get(s.lang_code, "You can enter the number or the name.")
            return f"{prompt}\n{numbered}\n\n{hint}", False, s.lang_code

        # 2) Auswahl verarbeiten (unverändert)
        if s.awaiting_selection and user_text:
            text = (user_text or "").strip()

            # a) Zahl?
            if text.rstrip(".").isdigit():
                idx = int(text.rstrip(".")) - 1
                if 0 <= idx < len(s.available_form_keys):
                    s.selected_form_key = s.available_form_keys[idx]
                    s.awaiting_selection = False
                    confirm = {
                        "de":"Verstanden. Wir starten mit dem Formular:",
                        "en":"Got it. We'll start with the form:",
                        "fr":"Compris. Nous commençons avec le formulaire :",
                        "tr":"Anlaşıldı. Şu form ile başlıyoruz:"
                    }.get(s.lang_code, "OK. We'll start with the form:")
                    return f"{confirm} **{s.translated_labels[idx]}**", True, s.lang_code

            # b) Exakter Text-Match (übersetzte Labels)
            lowered = text.lower()
            for i, lab in enumerate(s.translated_labels):
                if lowered == lab.lower():
                    s.selected_form_key = s.available_form_keys[i]
                    s.awaiting_selection = False
                    confirm = {
                        "de":"Verstanden. Wir starten mit dem Formular:",
                        "en":"Got it. We'll start with the form:",
                        "fr":"Compris. Nous commençons avec le formulaire :",
                        "tr":"Anlaşıldı. Şu form ile başlıyoruz:"
                    }.get(s.lang_code, "OK. We'll start with the form:")
                    return f"{confirm} **{lab}**", True, s.lang_code

            # c) Ungültig -> erneut anzeigen
            retry = {
                "de":"Ungültige Auswahl. Bitte wählen Sie erneut.",
                "en":"Invalid choice. Please choose again.",
                "fr":"Choix invalide. Veuillez recommencer.",
                "tr":"Geçersiz seçim. Lütfen tekrar seçin."
            }.get(s.lang_code, "Invalid choice. Please choose again.")
            numbered = self._format_numbered_list(s.translated_labels)
            return f"{retry}\n{numbered}", False, s.lang_code

        # Fallback
        return "…", False, s.lang_code

    def export_state(self) -> Dict[str, Any]:
        return {
            "turns": self.state.turns,
            "lang_code": self.state.lang_code,
            "available_form_keys": self.state.available_form_keys,
            "translated_labels": self.state.translated_labels,
            "awaiting_selection": self.state.awaiting_selection,
            "selected_form_key": self.state.selected_form_key,
            "conversation_id": self.state.conversation_id,
            "previous_response_id": self.state.previous_response_id,
        }
    

@dataclass
class ActivityWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    phase: str = "ask_have_desc"  # ask_have_desc | await_desc | llm_qa | confirm
    max_questions: int = 8
    transcript: List[Dict[str, str]] = field(default_factory=list)  # [{"role":"user"/"assistant","content": "..."}]
    final_activity_text: Optional[str] = None
    previous_response_id: Optional[str] = None

class ActivityWizard:
    """
    LLM-geführter Wizard:
    1) Nutzer hat schon eine Beschreibung? -> Prüfen/verbessern/Bestätigen.
    2) Sonst adaptive Q&A: Das LLM stellt jeweils EINE gezielte Frage, basierend auf dem bisherigen Kontext.
       - Stoppt spätestens nach max_questions.
       - Danach erzeugt es 1-2 Drafte und wählt den besten, validiert und gibt final zurück.
    """
    def __init__(self, state: Optional[ActivityWizardState] = None, model: str = "gpt-4o-mini"):
        self.state = state or ActivityWizardState()
        self.model = model
        self.client = OpenAI()

    # --------- Systemprompt (mehrsprachig, deutsch als Default) ----------
    def _system_prompt(self) -> str:
        return (
    "Du bist ein Assistent, der beim Ausfüllen des Feldes „Tätigkeit“ (Tätigkeitsbeschreibung) für ein deutsches "
    "Gewerbeanmeldungsformular unterstützt. Du führst ein KURZES, ZUSTANDSBEHAFTETES Interview in der vom Nutzer erkannten Sprache.\n\n"
    "ZIELE\n"
    "1) Stelle höchstens N gezielte Rückfragen (jeweils nur eine pro Schritt), wobei N = max_questions (separat vorgegeben).\n"
    "2) Jede Frage muss Unsicherheit reduzieren: Frage nur nach fehlenden Schlüsselinformationen.\n"
    "3) Sobald genügend Informationen gesammelt wurden ODER das Fragenbudget erreicht ist, generiere eine finale Tätigkeitsbeschreibung.\n\n"
    "QUALITÄTSREGELN\n"
    "- Befolge das Muster: „Tätigkeits-Art“ + „Tätigkeits-Objekt“ (+ „Tätigkeits-Ergänzung“).\n"
    "- Vermeide zu allgemeine Formulierungen wie „Handel mit Waren aller Art“.\n"
    "- Wenn mehrere Tätigkeiten genannt werden, markiere die Haupttätigkeit klar (verwende Unterstreichung mit Unterstrichen: _Haupttätigkeit_).\n"
    "- Sei spezifisch, aber nicht übermäßig eng gefasst (kleine zukünftige Geschäftserweiterungen sollen möglich bleiben).\n"
    "- Verwende 1–2 kurze Sätze; sachlich und klar formuliert.\n\n"
    "FRAGESTRATEGIE\n"
    "- Stelle pro Schritt nur EINE kurze Frage, in der Sprache des Nutzers.\n"
    "- Passe die Frage an die letzte Nutzerantwort an. Beispiel: Wenn der Nutzer „Bewachung“ angibt, frage nach, was bewacht wird "
    "(z. B. Gebäude, Veranstaltungen, Baustellen), stationär vs. mobil, bewaffnet/unbewaffnet, Zielgruppe (B2B/B2C), Region.\n"
    "- Für Handel: Großhandel/Einzelhandel/Online, welche Waren (Oberbegriff), Import/Export, Montage/Service?\n"
    "- Für Dienstleistungen: Bereich (z. B. Beratung, Montage, Reparatur), Gegenstand oder Thema, Auslieferungsform (online/offline, vor Ort), B2B/B2C.\n"
    "- Verwende lieber breite, aber legitime Oberbegriffe als erschöpfende Aufzählungen (z. B. „Elektrogeräte“ statt aller Marken).\n\n"
    "OUTPUT-API\n"
    "**HALTE DICH IMMER AN DAS VORGEGEBENE JSON-SCHEMA. GIB PRO SCHRITT GENAU EIN GÜLTIGES JSON-OBJEKT ZURÜCK!!!**\n"
    "Es gibt zwei Modi:\n"
    "- ask_next: Stelle die nächste, einzelne, bestmögliche Frage.\n"
    "- produce_draft: Erzeuge ein oder zwei Formulierungsvorschläge und gib eine gewählte, gültige finale Version zurück.\n\n"
    "VALIDIERUNG VOR produce_draft\n"
    "- Stelle sicher, dass die Formulierung nicht zu allgemein ist; lehne Phrasen wie „… aller Art“ ab.\n"
    "- Stelle sicher, dass die Haupttätigkeit eindeutig erkennbar ist, falls mehrere genannt werden."
)

    # --------- JSON Schemas for response formatting ----------
    def _schema_ask(self):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type":"string", "enum":["ask_next"]},
                "question":{"type":"string", "minLength":1}
            },
            "required":["mode","question"]
        }

    def _schema_draft(self):
        return {
            "type":"object",
            "additionalProperties": False,
            "properties":{
                "mode":{"type":"string","enum":["produce_draft"]},
                "candidates":{"type":"array","items":{"type":"string"}, "minItems":1, "maxItems":2},
                "final":{"type":"string","minLength":5},
                "notes":{"type":"string"}
            },
            "required": ["mode", "candidates", "final", "notes"]
        }

    # --------- LLM helpers ----------
    def _llm_check_and_improve(self, lang: str, text_in: str) -> Tuple[bool, str, str]:
        system = (
            "Validate and lightly improve a German 'Tätigkeit' description for a Gewerbeanmeldung. "
            "Keep the user's language; avoid '... aller Art'; keep concise and allow small future extensions; "
            "underline main activity with underscores if multiple."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "valid": {"type": "boolean"},
                "improved": {"type": "string"},
                "tips": {"type": "string"},
            },
            "required": ["valid", "improved", "tips"],
        }
        user = json.dumps({"lang": lang, "input": text_in}, ensure_ascii=False)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},  # <— STRING
            ],
            store=True,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "valid",
                    "schema": schema,
                    "strict": True,
                }
            },
            previous_response_id=self.state.previous_response_id,
        )
        self.state.previous_response_id = getattr(resp, "id", None)
        data = self._extract_first_json(resp.output_text)
        return bool(data["valid"]), data["improved"].strip(), data["tips"].strip()


    def _llm_next(self, lang: str, max_questions: int) -> dict:
        """
        Fragt das Modell nach EINE(R) nächsten Frage oder erzeugt (bei Budget 0) direkt den Draft.
        Keine oneOf-Variante – immer genau ein Schema.
        """
        # kompakten Kontext bauen
        convo = [{"role": m["role"], "content": m["content"]} for m in self.state.transcript]
        asked = sum(1 for m in convo if m["role"] == "assistant")
        control = {"lang": lang, "asked": asked, "answered": sum(1 for m in convo if m["role"] == "user"), "max_questions": max_questions}
        user_payload = json.dumps({"control": control, "conversation": convo}, ensure_ascii=False)

        # Schema strikt wählen
        if asked < max_questions:
            schema = self._schema_ask()          # NUR ask_next erlaubt
            schema_name = "activity_ask_next"
        else:
            schema = self._schema_draft()        # NUR produce_draft erlaubt
            schema_name = "activity_produce_draft"

        # Responses API – gleich wie in deinem LanguageWizard (String-Content + text.format)
        print(schema)
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_payload},
            ],
            store=True,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            previous_response_id=self.state.previous_response_id,
            temperature=0.5
        )
        self.state.previous_response_id = getattr(resp, "id", None)
        return self._extract_first_json(resp.output_text)



    # ---------- Public step() ----------
    def step(self, user_text: Optional[str]) -> Tuple[str, bool, Optional[str]]:
        s = self.state
        lang = s.lang_code or "de"

        # Turn 0: fragen, ob schon eine Beschreibung existiert
        if s.turns == 0:
            s.turns = 1
            return ("Hast du bereits eine Tätigkeitsbeschreibung für die Gewerbeanmeldung? (Ja/Nein)", False, lang)

        # Branch: Nutzer hat Beschreibung?
        if s.phase == "ask_have_desc" and user_text:
            t = (user_text or "").strip().lower()
            yes = {"ja","j","yes","y","oui","evet","ok","okay","klar"}
            no  = {"nein","n","no","non","hayır","hayir","nope"}
            if t in yes:
                s.phase = "await_desc"
                return ("Bitte gib deine aktuelle Tätigkeitsbeschreibung ein.", False, lang)
            if t in no:
                s.phase = "llm_qa"
                s.transcript = []  # reset QA convo
                # Starte sofort mit erster LLM-Frage
                nxt = self._llm_next(lang, s.max_questions)
                if nxt.get("mode") == "ask_next":
                    q = nxt["question"].strip()
                    s.transcript.append({"role":"assistant","content":q})
                    return (q, False, lang)
                else:
                    # LLM meint schon genug Kontext zu haben (unwahrscheinlich bei Turn 1)
                    final = nxt.get("final") or (nxt.get("candidates") or [""])[0]
                    # Validate/improve:
                    valid, improved, _ = self._llm_check_and_improve(lang, final)
                    s.final_activity_text = improved if valid else final
                    s.phase = "confirm"
                    return (f"Passt diese Formulierung?\n\n**{s.final_activity_text}** (Ja/Nein)", False, lang)
            # unklar
            return ("Bitte antworte mit Ja oder Nein. Hast du schon eine Tätigkeitsbeschreibung?", False, lang)

        if s.phase == "await_desc" and user_text:
            valid, improved, tips = self._llm_check_and_improve(lang, user_text)
            s.final_activity_text = improved if improved else user_text
            s.phase = "confirm"
            prefix = "" if valid else ("Die Beschreibung ist noch zu allgemein/unklar. Hinweise:\n" + tips + "\n\n")
            return (prefix + f"Passt diese Formulierung?\n\n**{s.final_activity_text}** (Ja/Nein)", False, lang)

        if s.phase == "confirm" and user_text:
            t = (user_text or "").strip().lower()
            yes = {"ja","j","yes","y","oui","evet","ok","okay","klar"}
            no  = {"nein","n","no","non","hayır","hayir","nope"}
            if t in yes:
                return ("Alles klar – ich übernehme diese Tätigkeitsbeschreibung. ✅", True, lang)
            if t in no:
                s.phase = "await_desc"
                return ("Gerne – bitte gib deine gewünschte Formulierung ein.", False, lang)
            return (f"Passt diese Formulierung?\n\n**{s.final_activity_text}** (Ja/Nein)", False, lang)

        # --------- LLM-gestützte Q&A ---------
        if s.phase == "llm_qa":
            if user_text:
                # Nutzerantwort anhängen
                s.transcript.append({"role":"user","content": user_text.strip()})

            # Nächsten Schritt vom LLM holen (Frage oder Draft)
            nxt = self._llm_next(lang, s.max_questions)

            # A) Modell stellt die nächste Frage
            if nxt.get("mode") == "ask_next":
                question = nxt["question"].strip()
                # Budgetcheck & Safety: wenn bereits max erreicht, erzwinge Draft im nächsten Turn
                asked = len([m for m in s.transcript if m["role"]=="assistant"])
                if asked >= s.max_questions:
                    # Force draft immediately
                    nxt = {"mode":"produce_draft", "final":"", "candidates":[]}
                else:
                    s.transcript.append({"role":"assistant","content": question})
                    return (question, False, lang)

            # B) Draft erzeugen (vom Modell initiiert oder wegen Max N)
            if nxt.get("mode") == "produce_draft":
                # Falls Kandidaten vorhanden, nimm den 'final', sonst ersten Kandidaten
                final = nxt.get("final") or (nxt.get("candidates") or [""])[0]
                valid, improved, tips = self._llm_check_and_improve(lang, final)
                s.final_activity_text = improved if valid else final
                s.phase = "confirm"
                prefix = "" if valid else ("Hinweis zur Präzisierung:\n" + tips + "\n\n")
                return (prefix + f"Passt diese Formulierung?\n\n**{s.final_activity_text}** (Ja/Nein)", False, lang)

        # Fallback
        return ("…", False, lang)
    
    @staticmethod
    def _extract_first_json(text: str):
        text = re.sub(r'^\s*```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj

    def export_state(self) -> Dict[str, Any]:
        s = self.state
        return {
            "turns": s.turns,
            "lang_code": s.lang_code,
            "phase": s.phase,
            "max_questions": s.max_questions,
            "transcript": s.transcript,
            "final_activity_text": s.final_activity_text,
            "previous_response_id": s.previous_response_id
        }