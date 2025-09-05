import requests
import logging
from typing import Optional
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ResponseFormat(BaseModel):
    input_message:str # User Input
    validity:str # True if input is valid, false otherwise
    corrected_input:str # Set if llm is told to correct input based on some rule, else empty
    in_valid_reason:str # reson for invalidity, if valid, empty


class LLMValidatorService:
    """
    Service class for sending prompts to a local LLM endpoint and retrieving responses.
    """

    @staticmethod
    def validate_locally(prompt: str, endpoint: str, max_tokens: int = 5,
                         temperature: float = 0.0, timeout: int = 10) -> Optional[str]:
        """
        Sends a prompt to the local LLM endpoint and retrieves the response content.
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "").strip()
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM local request failed: {e}")
            return None

    @staticmethod
    def validate_openai(prompt: str, model: str, client: OpenAI) -> Optional[str]:
        """
        Sends a prompt to the OpenAI API and retrieves the response.
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.choices or not response.choices[0].message:
                return None
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logger.error(f"OpenAI request failed: {e}")
            return None
        
    @staticmethod
    def validate_openai_json_mode(system_prompt: str, user_input:str, json_schema:dict, model: str, client: OpenAI) -> Optional[str]:
        """
        Sends a prompt to the OpenAI API and retrieves the response.
        """
        resp = client.responses.create(
        model=model,  # z.B. "gpt-4o-2024-08-06" oder "gpt-4o-mini"
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        text={
            "format": {
                "type": "json_schema",     # Structured Outputs aktivieren
                "name": "structured_output",# <-- MUSS hier stehen (nicht im Schema)
                "schema": json_schema,  # <-- reines JSON Schema-Objekt
                "strict": True             # harte Schemakonformität
            }
        }
    )
        return resp
            

    
