import datetime
import re
import os

from typing import List, Dict, Any, Union, Literal, Optional

from .llm_validator_service import LLMValidatorService
from .validator_helper import response_to_dict, convert_to_bool, load_txt
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
    def valid_not_empty(x:Union[str,int]) -> bool:
        '''
        Checks whether a filed is not empty
        '''
        if len(x.strip()) > 2:
            return True, "", x
        else:
            return False, "Dieses Feld darf nicht leer bleiben.", ""

    @staticmethod
    def valid_date(x:str) -> bool:
        """
        Dates have to be in the format TT.MM.JJJJ.
        """
        try:
            datetime.datetime.strptime(x, "%d.%m.%Y")
            return True, "", x
        except Exception:
            return False, "Bitte geben Sie das Datum im korrekten Format (TT.MM.JJJJ) an.", ""
        
    @staticmethod
    def valid_phone(x:str) -> bool:
        """
        Basic check, overwrite for more sophisticated check.
        """
        if bool(re.fullmatch(r"[+\d][\d\s\-/]{4,}", x)):
            return True, "", x
        else:
            return False, "Bitte geben Sie eine gültige Telefonnummer an (mindestens 5 Ziffern, kann Leerzeichen, +, - und / enthalten).", ""
    
    
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
    # super.__init__()
    # def valid_activity(self, x: str, llm_service=None) -> tuple:
    #     """
    #     Prüft, ob Tätigkeitsbeschreibung hinreichend präzise und zulässig ist.
    #     Rückgabe: (valid: bool, reason: str, payload: str)
    #     """
    #     check_prompt = (
    #         "Beispiele:\n\n"
    #         "Handel mit Waren aller Art – INVALID\n"
    #         "Herstellung von Kinderspielwaren – VALID\n"
    #         "Dienstleistungen aller Art – INVALID\n"
    #         "Selbstständigkeit im Bereich Liefer- und Kurierdienste – VALID\n"
    #         "Dinge verkaufen – INVALID\n"
    #         "Sanitärdienstleistungen – VALID\n"
    #         "Allgemeine Dienstleistungen – INVALID\n"
    #         "Großhandel mit Elektrowaren – VALID\n"
    #         "Chemische Kastration von Menschen – INVALID\n"
    #         "Auftragsmord - INVALID\n"
    #         "Import und Export von Menschen - INVALID\n"
    #         "Verkauf von Betäubungsmitteln an Privatpersonen - INVALID\n"
    #         "Online Marketing – INVALID\n\n"
    #         "Aufgabe:\n"
    #         "Prüfe, ob die folgende Tätigkeitsbeschreibung hinreichend präzise und zulässig ist (validity). Menschenverachtende oder Verbotene Tätigkeiten sind nicht zulässig (INVALID).\n"
    #         "Antwort ausschließlich mit: VALID (präzise genug & zulässig) oder INVALID (zu allgemein oder unzulässig).\n\n"
    #         f"Beschreibung: {x}\nAntwort:"
    #     )
    #     if llm_service is None:
    #         llm_service = LLMValidatorService()

    #     response = llm_service.validate_openai(
    #         prompt=check_prompt,
    #         model="gpt-4.1-mini",
    #         client=self.client
    #     )
    #     if not response:
    #         return False, "Keine Antwort vom LLM", x

    #     first_word = response.split()[0].upper()
    #     valid = (first_word == "VALID")
    #     reason = "" if valid else "Die Beschreibung ist nicht zulässig oder zu allgemein. Bitte 'Tätigkeits-Art' + 'Objekt' (+ 'Ergänzung') angeben und zu breite Formulierungen vermeiden."
    #     payload = x
    #     print(valid)
    #     # If valid, further check if a permit may be required
    #     if valid:
    #         print("check permit")
    #         needs_permit, permit_reason, _ = self.check_if_permit_is_required(x)
    #         if needs_permit == 'VALID':
    #             reason = permit_reason + ' Weitere Informationen finden Sie hier: <a href=\"https://www.ihk.de/konstanz/recht-und-steuern/gewerberecht/einzelne-berufe/erlaubnispflichtigegewerbe13180-1672696 \" target=\"_blank\">hier</a>.'



    #     return valid, reason, payload
    
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
            "Chemische Kastration von Menschen – INVALID\n"
            "Auftragsmord - INVALID\n"
            "Import und Export von Menschen - INVALID\n"
            "Verkauf von Betäubungsmitteln an Privatpersonen - INVALID\n"
            "Wartung und Betrieb von Kerntechnischen Anlagen - VALID\n"
            "Online Marketing – INVALID\n\n"
            "Aufgabe:\n"
            "Prüfe, ob die folgende Tätigkeitsbeschreibung hinreichend präzise und zulässig ist (validity). Menschenverachtende oder Verbotene Tätigkeiten sind nicht zulässig (INVALID).\n"
            "Antwort mit: VALID (präzise genug & zulässig) oder INVALID (zu allgemein oder unzulässig). Falls INVALID gib eine kurze prägnante Begründung an warum die Tätigkeitsbeschreibung INVALID ist ('Die Beschreibung ist leider ungültig weil/ da diese zu allgemein gefasst/ verboten/ menschverachtend ist.') (reason)\n\n"
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
        reason = "" if valid else response.output_parsed.reason + " Bitte 'Tätigkeits-Art' + 'Objekt' (+ 'Ergänzung') angeben und zu breite Formulierungen vermeiden."
        payload = x
        # If valid, further check if a permit may be required
        if valid:
            print("check permit")
            needs_permit, permit_reason, _ = self.check_if_permit_is_required(x)
            if needs_permit == 'VALID':
                reason = permit_reason + ' Weitere Informationen finden Sie hier: <a href=\"https://www.ihk.de/konstanz/recht-und-steuern/gewerberecht/einzelne-berufe/erlaubnispflichtigegewerbe13180-1672696 \" target=\"_blank\">hier</a>.'



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


    
    def valid_nationality(self,x):

        check_prompt = (
            "Aufgabe:\n"
            "Klassifiziere ob es sich bei {x} um eine VALIDE Stattsangehörikeit handelt.\n"
            "VALIDE ist Sie, wenn das Land tatsächlich existiert und korrekt geschrieben wurde. Anderenfalls ist Sie INVALID \n"
            "Antwort ausschließlich mit: VALID (korrekt geschrieben, tatsächliches Land) oder INVALID (falsch geschrieben, ausgedachtes Land).\n\n"
            f"Beschreibung: {x}\nAntwort:"
        )
        if llm_service is None:
            llm_service = LLMValidatorService()

        response = llm_service.validate_openai(
            prompt=check_prompt,
            model = "gpt-4.1-mini",
            client = self.client)
        if not response:
            return False
        first_word = response.split()[0].upper()
        return first_word == "VALID"
    
    def valid_registered_name(self,x):
        return self.valid_name(x)
    
    def valid_commercial_register_number(self,x):
        return self.valid_not_empty(x)
    
    # def valid_company_name(self,x):
    #     return self.valid_name(x)
    
    def valid_family_name(self,x):
        return self.valid_name(x)
    
    def valid_given_name(self,x):
        return self.valid_name(x)
    
    def valid_address(self,x):
        return self.valid_full_adress(x)

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

        if validity == 'VALID':
            adress = f"{response['street_name']}, {response['street_number']}, {response['postal_code']}, {response['city_name']}"
        else:
            adress = None

        return validity, reason, adress
    
    def valid_address(self, x, llm_service = None):
        return self.valid_representative_address(x, llm_service)
    
    def valid_main_branch_address(self, x, llm_service = None):
        return self.valid_representative_address(x, llm_service)

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
