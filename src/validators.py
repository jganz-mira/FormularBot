import datetime
import re
import os

from typing import List, Dict, Any, Union

from .llm_validator_service import LLMValidatorService
from openai import OpenAI

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8080/completion")

class BaseValidators:
    '''This class holds the most basic validation functions, each more specific 
    validators class can inherit from this class'''
    @staticmethod
    def valid_name(x:str) -> bool:
        '''
        Checks whether name has at least three characters.
        '''
        return isinstance(x, str) and len(x.strip()) > 2
    
    @staticmethod 
    def valid_not_empty(x:Union[str,int]) -> bool:
        '''
        Checks whether a filed is not empty
        '''
        return len(x.strip()) > 2

    @staticmethod
    def valid_date(x:str) -> bool:
        """
        Dates have to be in the format TT.MM.JJJJ.
        """
        try:
            datetime.datetime.strptime(x, "%d.%m.%Y")
            return True
        except Exception:
            return False
        
    @staticmethod
    def valid_phone(x:str) -> bool:
        """
        Basic check, overwrite for more sophisticated check.
        """
        return bool(re.fullmatch(r"[+\d][\d\s\-/]{4,}", x))
    
    @staticmethod
    def valid_adresse(x:str)-> bool:
        """
        Basic check, overwrite for more sophisticated check.
        """
        return isinstance(x, str) and len(x.strip()) > 7
    
    @staticmethod
    def valid_full_adress(x:str) -> bool:
        """
        Validates German addresses in the format:
        Straße, Hausnummer, Postleitzahl, Ort
        e.g. "Musterstraße, 12, 12345, Musterstadt"
        """
        pattern = re.compile(
            r'^'                              # start of string
            r'[A-ZÄÖÜ]'                       # street starts with capital letter
            r'[A-Za-zäöüÄÖÜß\s\.-]+'          # rest of street (letters, spaces, dot, hyphen)
            r',\s*'                           # comma + optional whitespace
            r'\d+'                            # house number (one or more digits)
            r'[A-Za-z]?'                      # optional single letter
            r',\s*'                           # comma + optional whitespace
            r'\d{5}'                          # five-digit postal code
            r',\s*'                           # comma + optional whitespace
            r'[A-ZÄÖÜ]'                       # city starts with capital letter
            r'[A-Za-zäöüÄÖÜß\s\.-]+'          # rest of city (letters, spaces, dot, hyphen)
            r'$'                              # end of string
        )
        return bool(pattern.fullmatch(x))
    
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
    def valid_activity(self, x: str, llm_service=None) -> bool:
        """
        Uses an LLM running at a configurable endpoint to perform ICL classification.

        Args:
            x (str): The occupation description to validate.
            llm_service (LLMValidatorService, optional): An instance of the LLMValidatorService. 
                If not provided, a default instance is used.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Default is 5.
            temperature (float, optional): Sampling temperature for the LLM. Default is 0.0 (deterministic).

        Returns:
            bool: True if the occupation type is sufficiently precise (VALID), False otherwise.
        """
        # Prompt for ICL. Add new examples as needed.
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
            "Online Marketing – INVALID\n\n"
            "Aufgabe:\n"
            "Prüfe, ob die folgende Tätigkeitsbeschreibung hinreichend präzise ist.\n"
            "Antwort ausschließlich mit: VALID (präzise genug) oder INVALID (zu allgemein).\n\n"
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

