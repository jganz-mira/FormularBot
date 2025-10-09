"""wizards are mainly short, stateful multiturn conversations with an llm.
If a user started a wizard in a former chat bot function call, the wizard is marked as active in the chatbot functions state and the information
needed to perform the wizards task is stored in a wizard handles object"""

# language_wizard.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import json
from openai import OpenAI
import json, re
from .translator import translate_from_de, instruction_msgs

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
            "1) Decide in which language the USER WANTS TO COMMUNICATE. Infer from the useres message. "
            "2) Return a short confirmation question in that language asking the user if they want to continue in that language."
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

    # def _llm_check_approval(self, user_text: str) -> Optional[bool]:
    #     """LLM-Backup für Ja/Nein-Klassifikation in beliebiger Sprache."""
    #     schema = {
    #         "type":"object",
    #         "additionalProperties": False,
    #         "properties":{
    #             "approved":{"type":"boolean"},
    #             "confirmation_prompt":{"type":"string"}},
    #         "required":["approved","confirmation_prompt"]
    #     }
    #     resp = self.client.responses.create(
    #         model=self.model,
    #         input=[
    #             {"role":"system","content":"Classify if the user message is a YES/approval or NO/rejection. If approval, formulate a short message with a warning like '\nPlease note: Your input will be translated to German for form filling. Please check the final form carefully before submission. (confirmation_prompt)."},
    #             {"role":"user","content":user_text}
    #         ],
    #         store=True,
    #         text={"format":{"type":"json_schema","name":"yn","schema":schema,"strict":True}},
    #         previous_response_id=self.state.previous_response_id
    #     )
    #     out = json.loads(resp.output_text)
    #     return bool(out.get("approved")), out.get("confirmation_prompt")

    def _llm_check_approval(self, user_text: str, last_assistant_msg: Optional[str] = None) -> Optional[bool]:
        """LLM-Backup für Ja/Nein-Klassifikation in beliebiger Sprache."""
        schema = {
            "type":"object",
            "additionalProperties": False,
            "properties":{
                "approved":{"type":"boolean"},
                "confirmation_prompt":{"type":"string"}},
            "required":["approved","confirmation_prompt"]
        }

        messages = [
            {"role":"system","content":"Classify if the user message is a YES/approval or NO/rejection. "
                                    "If approval, formulate a short message with a warning like '\nPlease note: "
                                    "Your input will be translated to German for form filling. "
                                    "Please check the final form carefully before submission. (confirmation_prompt)."}
        ]

        if last_assistant_msg:
            messages.append({"role": "assistant", "content": last_assistant_msg})

        messages.append({"role": "user", "content": user_text})

        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            store=True,
            text={"format":{"type":"json_schema","name":"yn","schema":schema,"strict":True}},
            previous_response_id=self.state.previous_response_id
        )
        out = json.loads(resp.output_text)
        return bool(out.get("approved")), out.get("confirmation_prompt")
    
    def _normalize(self, text: str) -> str:
        return (text or "").strip().lower()

    def _fast_language_from_text(self, user_text: str) -> Optional[str]:
        t = self._normalize(user_text)  # z.B. lowercasing, Trim etc.

        # Schlüsselwörter / Phrasen je Sprache
        de_keys = {
            "deutsch", "auf deutsch", "sprich deutsch", "german", "in german",
            "bitte deutsch", "deutsche sprache",
            # ISO-Code nur wenn alleinstehend (s. Logik unten): "de"
        }
        en_keys = {
            "englisch", "english", "speak english", "in english",
            "please english", "anglo", "eng", "eng language",
            # "en" nur wenn alleinstehend
        }
        fr_keys = {
            "französisch", "franzoesisch", "français", "francais",
            "en français", "in french", "french", "parlons français",
            # häufige französische Marker:
            "bonjour", "bonsoir", "salut", "est-ce que",
            # "fr" nur wenn alleinstehend
        }
        tr_keys = {
            "türkçe", "turkce", "turkish", "ingilizce degil türkçe", "türk dili",
            "türkisch", "auf türkisch", "in turkish",
            # "tr" nur wenn alleinstehend
        }

        # Tokenisiere in Wörter (Unicode-fähig), für exakte Wort-Matches
        tokens = set(re.findall(r"[^\W_]+", t, flags=re.UNICODE))  # Buchstaben-/Zahlen-„Wörter“ ohne _

        # Phrasen- oder Wort-Matcher (ganze Worte via Wortgrenzen)
        def has_any(keys: set[str]) -> bool:
            for k in keys:
                if " " in k:
                    # Phrase: als ganze Wörter suchen
                    if re.search(rf"\b{re.escape(k)}\b", t):
                        return True
                else:
                    # Einzelwort: exakter Wort-Match
                    if k in tokens:
                        return True
            return False

        # ISO-Codes akzeptieren wir nur, wenn sie allein stehen (z. B. Eingabe "de")
        iso_alone = {"de", "en", "fr", "tr"}
        is_only_iso = t.strip() in iso_alone and len(tokens) == 1

        if is_only_iso:
            return t.strip()

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
                    "de": "Alles klar – wir sprechen Deutsch. ✅\n",
                    "en": "Great — we'll continue in English. ✅\n**Please note: Your input will be translated to German for form filling. Please check the final form carefully before submission.**",
                    "fr": "Parfait — nous continuons en français. ✅\n**Veuillez noter : vos saisies seront traduites en allemand pour le remplissage du formulaire. Veuillez vérifier attentivement le formulaire final avant de le soumettre.**",
                    "tr": "Harika — Türkçe devam edelim. ✅\n**Lütfen dikkat: Girdiniz form doldurma için Almancaya çevrilecektir. Lütfen formu göndermeden önce dikkatlice kontrol edin.**",
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
            # approved, done_msg = self._llm_check_approval(user_text)
            last_assistant_msg = next((msg for role, msg in reversed(s.history) if role == "assistant"), None)

            approved, done_msg = self._llm_check_approval(user_text, last_assistant_msg)
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
        msg = "Auf **welcher Sprache** wollen wir uns unterhalten? Schreib einfach 'Deutsch', 'Englisch'...."
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
        prompt = data.get("prompt", "**Bitte wähle ein Formular aus:")
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
                prompt = "Bitte wähle ein Formular aus:"
                numbered = self._format_numbered_list(s.translated_labels)
                hint = "Gib dazu einfach die **Nummer** des angegebenen Formulars ein."
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
            }.get(s.lang_code, False)
            if not hint:
                hint = translate_from_de(text_de = "Sie können die Nummer oder den Namen eingeben.", target_lang=s.lang_code)
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
                    }.get(s.lang_code, False)
                    if not confirm:
                        confirm = translate_from_de(text_de = "Verstanden. Wir starten mit dem Formular:", target_lang=s.lang_code)
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
                    }.get(s.lang_code, False)
                    if not confirm:
                        confirm = translate_from_de(text_de = "Verstanden. Wir starten mit dem Formular:", target_lang=s.lang_code)
                    return f"{confirm} **{lab}**", True, s.lang_code

            # c) Ungültig -> erneut anzeigen
            retry = {
                "de":"Ungültige Auswahl. Bitte wählen Sie erneut.",
                "en":"Invalid choice. Please choose again.",
                "fr":"Choix invalide. Veuillez recommencer.",
                "tr":"Geçersiz seçim. Lütfen tekrar seçin."
            }.get(s.lang_code, False)
            if not retry:
                retry = translate_from_de(text_de = "Ungültige Auswahl. Bitte wählen Sie erneut.", target_lang=s.lang_code)
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

@dataclass
class ShortCutWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    phase: str = "ask_path"       # ask_path | capture | upload | cr_mock | review | ask_branch_addr | done
    choice: Optional[str] = None  # 'camera' | 'upload' | 'crf' | 'manual'
    extracted: Dict[str, Any] = field(default_factory=dict)
    edited: Dict[str, Any] = field(default_factory=dict)

class ShortCutWizard:
    """
    Fragt nach dem Abkürzungsweg:
      1) Foto aufnehmen (st.camera_input)
      2) Bild hochladen (st.file_uploader)
      3) Creditreform API (Mock)
      4) Manuell weiter

    Ablauf:
      - ask_path: 4 Buttons anzeigen (UI in streamlit_main.py)
      - capture/upload/cr_mock: Bild erfassen/hochladen oder Mock-Daten vorbereiten
      - review: Daten als Tabelle (st.data_editor) editieren, Buttons: "Übernehmen" / "Neues Bild"
      - ask_branch_addr: Ja/Nein zur Betriebsstätten-Anschrift
      - done: Wizard endet; Bot kann Slots füllen (Mapping-Hook in streamlit_main.py)
    """
    def __init__(self, state: Optional[ShortCutWizardState] = None):
        self.state = state or ShortCutWizardState()

    def step(self, user_text: Optional[str]) -> tuple[str, bool, Optional[str]]:
        s = self.state
        lang = s.lang_code or "de"

        if s.phase == "ask_path":
            s.turns += 1
            msg = (
                "Um das Ausfüllen zu beschleunigen, können Sie ihren **Handelsregisterauszug** hochladen. Wie möchten Sie fortfahren?\n\n"
                "1) **Foto aufnehmen**\n"
                "2) **Bild hochladen**\n"
                "3) **PDF Datei hochladen**\n"
                "4) **Creditreform (Mock)**\n"
                "5) **Manuell weiter ausfüllen**\n\n"
                "_Bitte verwenden Sie die Buttons unter der Nachricht._"
            )
            return msg, False, lang

        if s.phase in ("capture", "upload", "cr_mock"):
            # UI übernimmt hier (Kamera/Upload/Mock). Wir geben nur Status aus.
            txt = {
                "capture": "Bitte nehmen Sie jetzt ein Foto Ihres Handelsregisterauszugs auf.",
                "upload":  "Bitte laden Sie eine Bilddatei Ihres Handelsregisterauszugs hoch.",
                "cr_mock": "Creditreform-Mock wird vorbereitet …",
            }[s.phase]
            return txt, False, lang

        if s.phase == "review":
            return ("Bitte prüfen/ergänzen Sie die erkannten Daten in der Tabelle unten und klicken Sie anschließend **Daten übernehmen** oder **Neues Bild verwenden**.", False, lang)

        if s.phase == "ask_branch_addr":
            return ("Bezieht sich die angegebene **Adresse** auch auf die **Betriebsstätte**? (Ja/Nein)", False, lang)

        if s.phase == "done":
            return ("Alles klar – ich übernehme die Daten und fülle die passenden Felder. ✅", True, lang)

        return ("…", False, lang)

    def export_state(self) -> Dict[str, Any]:
        s = self.state
        return {
            "turns": s.turns,
            "lang_code": s.lang_code,
            "phase": s.phase,
            "choice": s.choice,
            "extracted": s.extracted,
            "edited": s.edited,
        }
    
    def apply_mapping_and_finish(self, app_state: dict, slots_def: list[dict]) -> None:
        """
        Übernimmt self.state.edited in app_state['responses'], aber nur, wenn der Slot existiert.
        Setzt anschließend den Wizard außer Kraft und signalisiert dem Bot, mit dem 1. Slot weiterzumachen.
        """
        # --- Guards & Vorbereitung
        if not isinstance(app_state, dict):
            return

        responses = app_state.setdefault("responses", {})
        slot_names = {s.get("slot_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        filed_names = {s.get("slot_name"):s.get("filed_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        edited = dict(self.state.edited or {})
        is_branch_same = bool(edited.get("_is_branch_addr_same"))

        def _has(slot: str) -> bool:
            return slot in slot_names

        def _val_from_edited(key: str, default=""):
            v = edited.get(key, default)
            return v.strip() if isinstance(v, str) else v

        def _set(slot: str, target_filed_name:str, value):
            """Minimal-schreibweise: {'value': value} (target_filed_name ergänzt der Bot später)"""
            if not _has(slot):
                return
            # In deinem Bot werden Responses als Dict mit 'value' erwartet.
            responses[slot] = {"value": value, "target_filed_name": target_filed_name}

        # --- 1) Direktes Feld-Mapping (flache Schlüssel) -----------------------
        direct_mapping = [
            ("authority",    "hra_office"),
            ("hra_number",   "hra_number"),
            ("company_name", "registered_name"),
            ("legal_type",   "registered_type"),
            ("activity",     "activity"),
        ]
        for src_key, slot_name in direct_mapping:
            if _has(slot_name):
                _set(slot_name, filed_names[slot_name], _val_from_edited(src_key))

        # --- 2) Adressen --------------------------------------------------------
        if _has("representative_address"):
            if is_branch_same:
                if _has("representative_address"):
                    _set("representative_address", filed_names.get("representative_address"), _val_from_edited("address"))
                    if _has("main_branch_address"):
                        _set("main_branch_address",filed_names.get("main_branch_address"), "")
            else:
                if _has("main_branch_address"):
                    _set("main_branch_address", filed_names.get("main_branch_address"), _val_from_edited("address"))
            # else:
            #     # leer lassen → Bot fragt normal weiter
            #     _set("main_branch_address", "")

        # --- 3) CEOs: erster Eintrag in Personenfelder + Anzahl -----------------
        # Erwartet: edited["ceo"] als List[Dict] mit Keys: family_name, given_name, city, birth_date
        ceo_data = edited.get("ceo")
        first_ceo = None
        ceo_count = 0

        # Normalisieren auf List[Dict]
        if isinstance(ceo_data, list):
            if ceo_data and isinstance(ceo_data[0], dict):
                # Bereits im neuen Schema
                ceo_list = ceo_data
            elif ceo_data and isinstance(ceo_data[0], list):
                # Alte Form: List[List[str]] → heben in Dicts
                ceo_list = []
                for row in ceo_data:
                    family = row[0] if len(row) > 0 else ""
                    given  = row[1] if len(row) > 1 else ""
                    city   = row[2] if len(row) > 2 else ""
                    bdate  = row[3] if len(row) > 3 else ""
                    ceo_list.append({
                        "family_name": family or "",
                        "given_name":  given or "",
                        "city":        city or "",
                        "birth_date":  bdate or "",
                    })
            else:
                ceo_list = []
        else:
            ceo_list = []

        ceo_count = len(ceo_list)
        first_ceo = ceo_list[0] if ceo_list else None

        # Anzahl an CEOs → num_representatives (als String, da Slot 'text' ist)
        if _has("num_representatives") and ceo_count:
            _set("num_representatives", filed_names.get("num_representatives"), str(ceo_count))

        # Ersten CEO in Einzel-Slots mappen (nur wenn vorhanden)
        if first_ceo:
            fam = (first_ceo.get("family_name") or "").strip()
            giv = (first_ceo.get("given_name")  or "").strip()
            bdt = (first_ceo.get("birth_date")  or "").strip()

            # Format-Kosmetik fürs Datum (falls "YYYY-MM-DD" → "DD.MM.YYYY")
            if bdt and re.match(r"^\d{4}-\d{2}-\d{2}$", bdt):
                yyyy, mm, dd = bdt.split("-")
                bdt = f"{dd}.{mm}.{yyyy}"

            if _has("family_name") and fam:
                _set("family_name", filed_names.get("family_name"), fam)
            if _has("given_name") and giv:
                _set("given_name", filed_names.get("given_name"), giv)
            if _has("birth_date") and bdt:
                _set("birth_date", filed_names.get("birth_date"), bdt)

        # --- 4) Wizard deaktivieren & Bot weiterschalten ------------------------
        app_state["active_wizard"] = "idcard_wizard"
        app_state["wizard_handles"] = None

        # # Suche von vorne beginnen, damit next_slot_index alle Prefills sauber überspringt
        # app_state["idx"] = 0

        # Signal: der Bot soll den nächsten offenen Slot sofort fragen
        app_state["awaiting_first_slot_prompt"] = False


@dataclass
class IDCardWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    phase: str = "ask_path"       # ask_path | capture | upload | review | done
    choice: Optional[str] = None  # 'camera' | 'upload' | 'manual'
    extracted: Dict[str, Any] = field(default_factory=dict)
    edited: Dict[str, Any] = field(default_factory=dict)

class IDCardWizard:
    """
    Fragt danach, ob der Ausweis zum Schnelleren Ausfüllen genutzt werden soll

    Ablauf:
      - ask_path: 2 Buttons anzeigen (UI in streamlit_main.py)
      - capture/upload: Bild erfassen/hochladen oder Mock-Daten vorbereiten
      - review: Daten als Tabelle (st.data_editor) editieren, Buttons: "Übernehmen" / "Neues Bild"
      - done: Wizard endet; Bot kann Slots füllen (Mapping-Hook in streamlit_main.py)
    """
    def __init__(self, state: Optional[IDCardWizardState] = None):
        self.state = state or IDCardWizardState()

    def step(self, user_text: Optional[str]) -> tuple[str, bool, Optional[str]]:
        s = self.state
        lang = s.lang_code or "de"

        if s.phase == "ask_path":
            s.turns += 1
            msg = (
                "Möchten Sie ein Foto von der Vorder- und Rückseite Ihres Ausweises hochladen um die restlichen Felder schneller auszufüllen?\n\n"
                "_Bitte verwenden Sie die Buttons unter der Nachricht._"
            )
            return msg, False, lang

        if s.phase in ("upload"):
            # UI übernimmt hier (Kamera/Upload/Mock). Wir geben nur Status aus.
            txt = {
                "upload":  "Bitte laden Sie eine Bilddatei der Vorder- und Rückseite Ihres Ausweises hoch.",
            }[s.phase]
            return txt, False, lang

        if s.phase == "review":
            return ("Bitte prüfen/ergänzen Sie die erkannten Daten in der Tabelle unten und klicken Sie anschließend **Daten übernehmen** oder **Neues Bild verwenden**.", False, lang)

        if s.phase == "done":
            return ("Alles klar – ich übernehme die Daten und fülle die passenden Felder. ✅", True, lang)

        return ("…", False, lang)

    def export_state(self) -> Dict[str, Any]:
        s = self.state
        return {
            "turns": s.turns,
            "lang_code": s.lang_code,
            "phase": s.phase,
            "choice": s.choice,
            "extracted": s.extracted,
            "edited": s.edited,
        }
    
    def apply_mapping_and_finish(self, app_state: dict, slots_def: list[dict]) -> None:
        """
        Übernimmt self.state.edited in app_state['responses'], aber nur, wenn der Slot existiert.
        Setzt anschließend den Wizard außer Kraft und signalisiert dem Bot, mit dem 1. Slot weiterzumachen.
        """
        # --- Guards & Vorbereitung
        if not isinstance(app_state, dict):
            return

        responses = app_state.setdefault("responses", {})
        slot_names = {s.get("slot_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        filed_names = {s.get("slot_name"):s.get("filed_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        edited = dict(self.state.edited or {})

        def _has(slot: str) -> bool:
            return slot in slot_names

        def _val_from_edited(key: str, default=""):
            v = edited.get(key, default)
            return v.strip() if isinstance(v, str) else v

        def _set(slot: str, target_filed_name:str, value):
            """Minimal-schreibweise: {'value': value} (target_filed_name ergänzt der Bot später)"""
            if not _has(slot):
                return
            
            if slot == 'nationality':
                responses[slot] = {"value": str(value), "target_filed_name": target_filed_name, "choices" : ["ja","nein"]}
            else:
                # In deinem Bot werden Responses als Dict mit 'value' erwartet.
                responses[slot] = {"value": value, "target_filed_name": target_filed_name}

        # --- 1) Direktes Feld-Mapping -----------------------
        for slot_name in edited:
            if _has(slot_name):
                _set(slot_name, filed_names[slot_name], _val_from_edited(slot_name))

        # --- 2) Wizard deaktivieren & Bot weiterschalten ------------------------
        app_state["active_wizard"] = None
        app_state["wizard_handles"] = None

        # Suche von vorne beginnen, damit next_slot_index alle Prefills sauber überspringt
        app_state["idx"] = 0

        # Signal: der Bot soll den nächsten offenen Slot sofort fragen
        app_state["awaiting_first_slot_prompt"] = True

@dataclass
class PreRegistrationWizardState:
    turns: int = 0
    lang_code: Optional[str] = None
    phase: str = "ask_start"   # ask_start -> ask_reg_for -> done
    edited: Dict[str, Any] = field(default_factory=dict)

class PreRegistrationWizard:
    """
    Kleiner Wizard vor dem Shortcut:
      1) Startdatum (start_date)
      2) Art der Niederlassung (registration_for)
    Danach: -> ShortcutWizard
    """
    def __init__(self, state: Optional[PreRegistrationWizardState] = None):
        self.state = state or PreRegistrationWizardState()

    def step(self, user_text: Optional[str]) -> tuple[str, bool, Optional[str]]:
        s = self.state
        lang = s.lang_code or "de"

        if s.phase == "ask_start":
            s.turns += 1
            return ("Wann **wirst** oder wann **hast** du mit deiner Tätigkeit begonnen?", False, lang)

        if s.phase == "ask_reg_for":
            return ("Für was für eine **Art von Niederlassung**, möchtest du die Anmeldung vornehmen?", False, lang)

        if s.phase == "done":
            return ("Alles klar 👍. Ich habe die Angaben übernommen, weiter gehts!", True, lang)

        return ("…", False, lang)

    def apply_mapping_and_finish(self, app_state: dict, slots_def: List[dict]) -> None:
        """
        Übernimmt start_date und registration_for in app_state['responses'].
        Schaltet anschließend auf den ShortcutWizard um.
        """
        if not isinstance(app_state, dict):
            return

        responses = app_state.setdefault("responses", {})
        slot_names = {s.get("slot_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        filed_names = {s.get("slot_name"): s.get("filed_name") for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}
        slot_by_name = {s.get("slot_name"): s for s in (slots_def or []) if isinstance(s, dict) and s.get("slot_name")}

        def _has(slot: str) -> bool:
            return slot in slot_names

        def _set(slot: str, value):
            if not _has(slot):
                return
            payload = {"value": value, "target_filed_name": filed_names.get(slot)}
            # Falls der Slot Choices hat (Checkboxen/Radio), mitgeben → PDF-Check passt dann robust
            if "choices" in (slot_by_name.get(slot) or {}):
                payload["choices"] = slot_by_name[slot]["choices"]
            responses[slot] = payload

        edited = dict(self.state.edited or {})
        if _has("start_date") and edited.get("start_date"):
            _set("start_date", edited["start_date"])

        if _has("registration_for") and edited.get("registration_for"):
            _set("registration_for", edited["registration_for"])

        # → jetzt direkt den ShortcutWizard aktivieren
        app_state["active_wizard"] = "shortcut_wizard"
        app_state["wizard_handles"] = None