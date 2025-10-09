import datetime
import re
import os

from typing import List, Dict, Any, Union, Literal, Optional

from .llm_validator_service import LLMValidatorService
from .validator_helper import response_to_dict, convert_to_bool, load_txt, is_gp_town
from openai import OpenAI
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+
from pydantic import BaseModel


LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8080/completion")

class PermitSchema(BaseModel):
    validity: Literal["VALID", "INVALID"]
    permit_reason: Optional[str] = None

class ActivityCheckResponse(BaseModel):
    validity: Literal["VALID", "INVALID"]
    reason: Optional[str] = None

class BaseValidators:
    '''This class holds the most basic validation functions, each more specific 
    validators class can inherit from this class'''

    @staticmethod
    def valid_basic(x:str) -> tuple:
        """
        does not check anything, only here for convinience. Is used if not validation is available
        """
        return True, "", x

    @staticmethod
    def valid_name(x:str) -> bool:
        '''
        Checks whether name has at least three characters.
        '''

        validity = isinstance(x, str) and len(x.strip()) > 2

        reason = '' if validity else "Der Name muss mindestens drei Zeichen lang sein."

        payload = x if validity else ''

        return validity, reason, payload
    
    @staticmethod 
    def valid_not_empty(x:Union[str,int], min_length = 2) -> bool:
        '''
        Checks whether a filed is not empty
        '''
        if len(x.strip()) > min_length:
            return True, "", x
        else:
            return False, "Dieses Feld darf nicht leer bleiben.", ""

    @staticmethod
    def valid_date(x:str) -> bool:
        """
        Dates have to be in the format TT.MM.JJJJ.
        """
        try:
            datetime.strptime(x, "%d.%m.%Y")
            return True, "", x
        except Exception:
            return False, "Bitte geben Sie das Datum im korrekten Format (TT.MM.JJJJ) an.", ""
        
    @staticmethod
    def valid_phone(x:str) -> bool:
        """
        Basic check, overwrite for more sophisticated check.
        """
        if bool(re.fullmatch(r"[+\d][\d\s\-/]{4,}", x)):
            return True, f"Die Telfonnummer {x} wird eingetragen.", x
        else:
            return False, "Bitte geben Sie eine gültige Telefonnummer an (mindestens 5 Ziffern, kann Leerzeichen, +, - und / enthalten).", ""

    @staticmethod
    def valid_email(x: str) -> tuple[bool, str, str]:
        """
        Basic email check, overwrite for more sophisticated check.
        """
        if bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", x)):
            return True, f"Die E-Mail-Adresse {x} wird eingetragen.", x
        else:
            return False, "Bitte geben Sie eine gültige E-Mail-Adresse im Format name@domain.tld an.", ""

    
    @staticmethod
    def valid_choice_slot(message: str, slot_def: Dict[str, Any]) -> bool:
        """
        Validates a 'choice'-slot by checking if the user's input corresponds
        to one of the defined choices.

        - If the input is a digit, it's interpreted as 1-based index into 'choices'.
        - Otherwise, it's compared (case-insensitive) to each choice string.

        Args:
            message (str): The raw user input.
            slot_def (dict): The slot definition dict, must contain "choices": List[str].

        Returns:
            bool: True if the input maps to one of the choices, False otherwise.
        """
        choices: List[str] = slot_def.get("choices", [])
        text = message.strip()
        
        # 1) Zifferneingabe als Index?
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(choices):
                return True
        
        # 2) Exakte Text-Übereinstimmung (case-insensitive)
        for opt in choices:
            if text.lower() == opt.lower():
                return True
        
        return False
    
class GewerbeanmeldungValidators(BaseValidators):
    def __init__(self):
        super().__init__()
        # initialize client and set model, theoretically, the model for each task could be set dynamically (use different models for more or less complex tasks)
        self.client = OpenAI()
        self.llm_service = LLMValidatorService()

    def valid_registered_name(self,x):
        return self.valid_name(x)
    
    def valid_start_date(self, x: str, llm_service=None) -> tuple:
        """
        Prüft ein Datum im Format TT.MM.JJJJ.
        Regeln:
          - Format muss exakt TT.MM.JJJJ sein (mit führenden Nullen).
          - Datum darf maximal 1 Monat (= 31 Tage) in der Vergangenheit liegen.
          - Zukünftige Daten sind erlaubt.
        Rückgabe: (valid: bool, reason: str, payload: str)
        """
        user_input = (x or "").strip()

        # 1) Formatprüfung
        m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", user_input)
        if not m:
            return (
                False,
                "Bitte geben Sie das Datum im Format TT.MM.JJJJ an (z. B. 05.09.2025).",
                user_input,
            )

        day, month, year = map(int, m.groups())

        # 2) Kalenderdatum validieren (z. B. 31.02. ist ungültig)
        try:
            date_val = datetime(year, month, day, tzinfo=ZoneInfo("Europe/Berlin")).date()
        except ValueError:
            return (
                False,
                "Ungültiges Datum (z. B. 31.02. existiert nicht). Bitte prüfen Sie Ihre Eingabe.",
                user_input,
            )

        # 3) Stichtag berechnen (heute in Europe/Berlin)
        today = datetime.now(ZoneInfo("Europe/Berlin")).date()
        one_month_ago = today - timedelta(days=31)  # robuste, einfache Definition

        # 4) Regel: maximal 1 Monat in der Vergangenheit erlaubt
        if date_val < one_month_ago:
            return (
                False,
                "Das Datum liegt mehr als einen Monat in der Vergangenheit. "
                "Bitte ein Datum wählen, das höchstens 1 Monat zurückliegt.",
                user_input,
            )

        # 5) OK → normalisierte Payload (TT.MM.JJJJ mit führenden Nullen)
        payload = f"{day:02d}.{month:02d}.{year:04d}"
        return True, "", payload

    def valid_activity(self, x: str, llm_service=None) -> tuple:
        """
        Prüft, ob Tätigkeitsbeschreibung hinreichend präzise und zulässig ist.
        Rückgabe: (valid: bool, reason: str, payload: str)
        """
        check_prompt = (
            "Beispiele:\n\n"
            "Handel mit Waren aller Art – INVALID\n"
            "Herstellung von Kinderspielwaren – VALID\n"
            "Dienstleistungen aller Art – INVALID\n"
            "Selbstständigkeit im Bereich Liefer- und Kurierdienste – VALID\n"
            "Dinge verkaufen – INVALID\n"
            "Sanitärdienstleistungen – VALID\n"
            "Allgemeine Dienstleistungen – INVALID\n"
            "Großhandel mit Elektrowaren – VALID\n"
            "Import/Export von Menschen – INVALID\n"
            "Auftragsmord – INVALID\n"
            "Verkauf von Betäubungsmitteln an Privatpersonen – INVALID\n"
            "Wartung und Betrieb von kerntechnischen Anlagen – VALID\n"
            "Import und Export von nicht verschreibungspflichtigen Medikamenten – VALID\n"
            "Online Marketing – INVALID\n\n"
            "Aufgabe:\n"
            "Prüfe, ob die folgende Tätigkeitsbeschreibung hinreichend präzise ist (validity).\n"
            "Eine Tätigkeitsbeschreibung ist INVALID nur wenn sie:\n"
            "a) zu allgemein ist (z. B. 'Dienstleistungen aller Art'), oder\n"
            "b) eindeutig menschenverachtend oder offensichtlich kriminell ist "
            "(z. B. Mord, Menschenhandel, Verkauf illegaler Drogen).\n\n"
            "Treffe KEINE Annahmen über gesetzliche Vorschriften oder mögliche "
            "Genehmigungspflichten. Beurteile nur nach Präzision und offensichtlicher "
            "ethischer Unzulässigkeit.\n\n"
            "Antwort mit: VALID (präzise genug & zulässig) oder INVALID (zu allgemein oder unzulässig).\n"
            "Falls INVALID, gib eine kurze Begründung an ('Die Beschreibung ist leider ungültig, weil ...').\n\n"
        )

        user_input = f"Beschreibung: {x}"

        if llm_service is None:
            llm_service = LLMValidatorService()

        response = llm_service.validate_openai_structured_output(
            system_prompt=check_prompt,
            user_input=user_input,
            model="gpt-4o-mini",
            client=self.client,
            json_schema = ActivityCheckResponse
        )
        if not response:
            return False, "Keine Antwort vom LLM", x

        valid = (response.output_parsed.validity == "VALID")
        reason = "" if valid else response.output_parsed.reason + " Ihre Formulierung sollte Tätigkeitsart, Tätigkeitsobjekt so wie gegebenenfalls Ergänzugen enthalten.Zu breite Formulierungen sollten allerdings vermieden werden."
        payload = x
        # If valid, further check if a permit may be required
        if valid:
            print("check permit")
            needs_permit, permit_reason, _ = self.check_if_permit_is_required(x)
            if needs_permit == 'VALID':
                reason = permit_reason + ' Weitere Informationen finden Sie [hier](https://www.ihk.de/konstanz/recht-und-steuern/gewerberecht/einzelne-berufe/erlaubnispflichtigegewerbe13180-1672696).'



        return valid, reason, payload
    
    def check_if_permit_is_required(self, x: str, llm_service=None) -> tuple:
        """ Takes the User inut and checks, whether a permit may be needed for the activity."""

        # load pdf document which contains a list of activities which require a permit
        permit_list = load_txt('./data/jobs_which_need_permit.txt')
        

        system_prompt = f"""Du bist ein Assistent, welcher Tätigkeitsbeschreibungen mit den in der folgenden Liste
                        definierten Berfusbezeichnungen abgleicht, und die Berufsbezeichnung als Erlaubnisbedürftig (VALID) oder nicht Erlaubnisbedürftig (INVALID) klassifiziert (validity).
                        Ist eine Berufsbezeichnung möglicherweise Erlaubnispflichtig, dann gib den Grund dafür in einem kurzen, erklärenden Satz an (permit_reason)
                        'Die angegebene Tätigkeit ist möglicherweise erlaubnispflichtig nach ....'.
                        Eine Tätigkeit muss nicht zwingend vollständig mit der 
                        Tätigkeitsbeschreibung übereinstimmen, sondern es reicht, wenn die Tätigkeit inhaltlich ähnlich ist. Wenn die Tätigkeit nicht erlaubnispflichtig ist, dann lasse das Feld
                        permit_reason leer. Antworte ausschließlich mit dem JSON-Objekt, ohne weitere Erklärungen. Du kannst die Informationen aus der Liste
                        für die Erklärung übernehmen, aber **unter keinen Umständen darfst du neue erfinden oder vorhandene Ändern. Das ist von höchster Priorität**.\n **Die Liste:**{permit_list}"""


        if llm_service is None:
            llm_service = LLMValidatorService()

        response = llm_service.validate_openai_structured_output(
            system_prompt = system_prompt,
            user_input = x,
            json_schema = PermitSchema,
            client = self.client,
            model = 'gpt-5-mini'
        )
        validity = response.output_parsed.validity
        reason = response.output_parsed.permit_reason

        # extrahiere validity nach validity, reason nach reason und payload bleibt leer
        return validity, reason, ''
    
    def valid_representative_address(self, x, llm_service = None) -> bool:
        system_prompt = (
            "Aufgabe: Extrahiere aus der Nutzereingabe den Straßennamen, die Hausnummer, die Postleitzahl und den Stadtnamen. "
            "Wenn alle Angaben vorhanden sind, gib 'VALID' zurück; wenn auch nur eine Information fehlt "
            "(z. B. keine Postleitzahl, keine Hausnummer, kein Stadtname, kein Straßenname), gib 'INVALID' zurück.\n"
            "Wenn du kleine Tippfehler im Stadtnamen findest, gib den korrigierten Namen in 'city_name' zurück.\n"
            "Falls 'INVALID', gib im Feld 'invalid_reason' eine kurze Begründung an, warum die Eingabe 'INVALID' ist "
            "(z. B. fehlende Hausnummer, fehlende Postleitzahl, falsche Postleitzahl ...).\n"
            "WICHTIG: Halte dich strikt an das angegebene JSON-Format. **Unter keinen Umständen fehlende Informationen erfinden.** "
            "Kein Freitext, keine Erklärungen – **das ist von höchster Wichtigkeit.** "
            "Eine gültige Postleitzahl muss eine deutsche Postleitzahl mit genau 5 Ziffern zwischen 01000 und 99999 sein."
        )

        address_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "validity": {
                    "type": "string",
                    "enum": ["VALID", "INVALID"],
                    "description": "VALID, wenn Eingabe vollständig und gültig; sonst INVALID"
                },
                "invalid_reason": {
                    "type": "string",
                    "description": "Grund für die Ungültigkeit; leer lassen, falls gültig"
                },
                "street_name": {
                    "type": "string",
                    "description": "Straßenname aus der Nutzereingabe; leer, falls fehlt"
                },
                "street_number": {
                    "type": "string",
                    "description": "Hausnummer aus der Nutzereingabe; leer, falls fehlt"
                },
                "postal_code": {
                    "type": "string",
                    "description": "Postleitzahl aus der Nutzereingabe; leer, falls fehlt"
                },
                "city_name": {
                    "type": "string",
                    "description": "Stadtname aus der Nutzereingabe; leer, falls fehlt"
                }
            },
            "required": [
                "validity",
                "invalid_reason",
                "street_name",
                "street_number",
                "postal_code",
                "city_name"
            ]
        }

        if llm_service is None:
            llm_service = LLMValidatorService()

        response = self.llm_service.validate_openai_json_mode(
            system_prompt=system_prompt,
            user_input=x,
            json_schema = address_schema,
            model="gpt-4.1-mini",
            client = self.client
        )

        response = response_to_dict(response)
        
        validity = convert_to_bool(response['validity'])
        reason = response['invalid_reason']
        print(validity)
        if validity:
            adress = f"{response['street_name']}, {response['street_number']}, {response['postal_code']}, {response['city_name']}"
        else:
            adress = None

        # # validate if in Göppingen
        # reason += is_gp_town(response['postal_code'])

        # print(adress)

        return validity, reason, adress

    def valid_rep_phone(self,x):
        if x == "":
            return True, "Telefonummer wird übersprungen", ""
        else:
            validity, reason, payload = self.valid_phone(x)
            return validity, reason, payload
        
    def valid_rep_email(self,x):
        if x == "":
            return True, "E-Mail wird übersprungen", ""
        else:
            validity, reason, payload = self.valid_email(x)
            return validity, reason, payload
        
    # def valid_main_branch_address(self, x, llm_service = None):
    #     return self.valid_representative_address(x, llm_service)
    
    def valid_main_branch_phone(self,x):
        if x == "":
            return True, "Telefonummer wird übersprungen", ""
        else:
            validity, reason, payload = self.valid_phone(x)
            return validity, reason, payload
        
    def valid_main_branch_email(self,x):
        if x == "":
            return True, "E-Mail wird übersprungen", ""
        else:
            validity, reason, payload = self.valid_email(x)
            return validity, reason, payload

    def valid_num_representatives(self, x: str):
        """
        Prüft, ob eine Nutzereingabe nicht leer ist und >= 1 liegt.
        Bonus: erkennt ausgeschriebene deutsche Zahlen von 1 bis 20,
        auch wenn sie in einem Satz oder mit anderen Wörtern kombiniert sind.
        Gibt als payload immer die Zahl (int) zurück.
        """
        if not x or not x.strip():
            return False, "Bitte geben Sie eine Zahl ein (mindestens 1).", ""

        # ausgeschriebene deutsche Zahlen 1–20
        words_to_nums = {
            "eins": 1, "eine": 1, "einer": 1, "einem": 1,
            "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
            "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
            "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13,
            "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
            "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
            "zwanzig": 20,
        }

        x_clean = x.strip().lower()

        # 1) Prüfen auf direkte Zahl im Text
        m = re.search(r"\d+", x_clean)
        if m:
            val = int(m.group())
            if val >= 1:
                return True, f"Die Anzahl der Vertreter ({val}) wird eingetragen.", str(val)
            else:
                return False, "Die Anzahl muss mindestens 1 sein.", ""

        # 2) Prüfen auf ausgeschriebenes Wort
        for word, val in words_to_nums.items():
            if re.search(rf"\b{word}\b", x_clean):
                return True, f"Die Anzahl der Vertreter ({val}) wird eingetragen.", str(val)

        return False, "Bitte geben Sie eine gültige Zahl (als Ziffer oder ausgeschrieben bis 20) ein.", ""
    
    def valid_num_partners(self, x: str):
        return self.valid_num_representatives(x)

    def valid_employees_full_time(self, x: str):
        """
        Prüft die Eingabe für 'employees_full_time':
        - nicht leer
        - Zahl >= 0
        - versteht ausgeschriebene deutsche Zahlen bis 20
        - payload = int
        """
        if not x or not x.strip():
            return False, "Bitte geben Sie die Zahl Ihrer Vollzeitkräfte an (mindestens 0).", ""

        words_to_nums = {
            "null": 0,
            "eins": 1, "eine": 1, "einer": 1, "einem": 1,
            "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
            "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
            "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13,
            "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
            "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
            "zwanzig": 20,
        }

        x_clean = x.strip().lower()

        # 1) Zahl als Ziffer
        m = re.search(r"\d+", x_clean)
        if m:
            val = int(m.group())
            if val >= 0:
                return True, f"Die Zahl der Vollzeitkräfte ({val}) wird eingetragen.", val
            else:
                return False, "Die Zahl muss mindestens 0 sein.", ""

        # 2) Zahl ausgeschrieben
        for word, val in words_to_nums.items():
            if re.search(rf"\b{word}\b", x_clean):
                return True, f"Die Zahl der Vollzeitkräfte ({val}) wird eingetragen.", val

        return False, "Bitte geben Sie eine gültige Zahl (als Ziffer oder ausgeschrieben bis 20) ein.", ""

    def valid_employees_part_time(self, x: str):
        return self.valid_employees_full_time(x)
    
    def valid_permit_date(self, x: str):
        return self.valid_date(x)
    
    def valid_permit_office(self, x: str):
        return self.valid_not_empty(x)
    
    def valid_handwerkskarte_date(self, x: str):
        return self.valid_date(x)
    
    def valid_handwerkskarte_office(self, x: str):
        return self.valid_not_empty(x)
    
    def valid_family_name(self,x):
        return self.valid_name(x)
    
    def valid_given_name(self,x):
        return self.valid_name(x)
    
    def valid_birth_name(self,x):
        if x != "":
            return self.valid_name(x)
        else:
            return True, "Das Feld wird übersprungen", ""
        
    def valid_birth_date(self,x):
        return self.valid_date(x)
    
    def valid_birth_place(self, x: str):
        """
        Prüft die Eingabe für 'birth_place_city':
        - darf nicht leer sein
        - sollte ein Komma enthalten (Format: Ort, Land)
        """
        if not x or not x.strip():
            return False, "Bitte geben Sie Ihren Geburtsort und das Land an (z. B. 'Berlin, Deutschland').", ""

        if "," not in x:
            return False, "Bitte geben Sie Ihren Geburtsort im Format 'Ort, Land' an.", ""

        return True, f"Geburtsort '{x.strip()}' wird eingetragen.", x.strip()
    
    def valid_other_nationality(self, x: str, llm_service=None):
        """
        Prüft, ob 'x' eine valide Staatsangehörigkeit / ein existierendes Land in korrekter Schreibweise ist.
        Nutzt LLM im JSON-Mode. Ergebnis-Felder:
        - validity: "VALID" | "INVALID"
        - invalid_reason: kurzer Grund (leer, wenn VALID)
        - country_name: normierter Ländername, falls VALID (sonst leer)
        - suggested_country: korrigierter Ländername bei leichten Tippfehlern (sonst leer)
        Rückgabe: (bool_valid, reason, payload_country)
        """
        system_prompt = (
            "Aufgabe:\n"
            "Klassifiziere, ob es sich bei der Eingabe um eine VALIDE Staatsangehörigkeit handelt.\n"
            "VALIDE ist sie NUR, wenn das Land tatsächlich existiert UND die Schreibweise größtenteils korrekt ist.\n"
            "Ist das Land ausgedacht oder nicht zu erkennen, um welches Land es sich handeln soll antworte mit INVALID und lasse 'country_name' leer.\n"
            "Wenn die Schreibweise NUR LEICHT falsch ist aber klar zu erkennen ist um welches Land es sich handelt, gib VALID zurück, gib im Feld 'country_name' den normierten offiziellen Ländernamen an.\n"
            "**Halte dich strikt an das JSON-Schema. Keine Erklärtexte außerhalb der Felder. Das ist von höchster Wichtigkeit.**\n"
        )

        nationality_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "validity": {
                    "type": "string",
                    "enum": ["VALID", "INVALID"],
                    "description": "VALID nur bei existierendem Land mit korrekter Schreibweise, sonst INVALID."
                },
                "country_name": {
                    "type": "string",
                    "description": "Normierter offizieller Ländername, NUR füllen, wenn VALID; sonst leer."
                }
            },
            "required": ["validity", "country_name"]
        }

        if llm_service is None:
            llm_service = LLMValidatorService()

        response = llm_service.validate_openai_json_mode(
            system_prompt=system_prompt,
            user_input=f"Beschreibung: {x}\nAntwort:",
            json_schema=nationality_schema,
            model="gpt-4.1-mini",
            client=self.client
        )

        # Hilfsfunktion, die dein Service vermutlich bereitstellt; andernfalls: json.loads(response)
        resp = response_to_dict(response)

        validity_str = resp.get("validity", "INVALID")
        is_valid = (validity_str == "VALID")
        if is_valid:
            return True, f"Die Staatsangehörigkeit '{resp.get('country_name','')}' wird eingetragen.", resp.get("country_name", x).strip()
        else:
            return False, f"Leider ist '{x}' keine gültige Staatsangehörigkeit.", ""

    def valid_residence_permit_date(self, x: str):
        return self.valid_date(x)
    
    def valid_residence_permit_office(self, x: str):
        return self.valid_not_empty(x)
    
    def valid_residence_permit_restriction_details(self, x: str):
        return self.valid_not_empty(x, min_length = 10)
    
    def valid_address(self,x):
        return self.valid_representative_address(x)
