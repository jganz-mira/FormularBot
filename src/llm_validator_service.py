import requests
import logging
from typing import Optional
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

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
        

    
