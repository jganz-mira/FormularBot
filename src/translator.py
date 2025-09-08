from typing import Optional
from openai import OpenAI

SUPPORTED = {"de","en","fr","tr"}

def translate_from_de(text_de: str, target_lang: str, client: Optional[OpenAI] = None, model: str = "gpt-4.1-mini") -> str:
    """
    Übersetzt 'text_de' von Deutsch -> target_lang (ISO-639-1).
    - Platzhalter {so_was} / {{so_was}} / <TAGS> bleiben unverändert.
    - Juristische Begriffe wie GmbH, AG, UG, OHG, KG, e.K. usw. niemals übersetzen.
    - Bei target_lang == 'de' oder unbekanntem Code -> Rückgabe = text_de (fail-safe).
    """
    tgt = (target_lang or "de").lower()
    if tgt not in SUPPORTED or tgt == "de" or not text_de:
        return text_de

    client = client or OpenAI()

    system_prompt = (
        "You are a precise translator. Translate from German into the target language.\n"
        f"- Target language (ISO 639-1): {tgt}\n"
        "- Keep placeholders and variables exactly as-is: {like_this}, {{like_this}}, <TAGS>, $VARS, %(fmt)s.\n"
        "- Do not translate URLs, emails, codes, or content inside {{double braces}}.\n"
        "- Do not translate legal/corporate terms such as GmbH, AG, UG, OHG, KG, e.K., mbH.\n"
        "- Preserve punctuation, line breaks, and Markdown.\n"
        "- Style: concise, polite, clear."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "input_text", "text": text_de}]},
        ],
    )

    return (resp.output_text or "").strip()

def translate_to_de(text_src: str, source_lang: str, client: Optional[OpenAI] = None, model: str = "gpt-4.1-mini") -> str:
    """
    Übersetzt 'text_src' von source_lang -> Deutsch.
    - Platzhalter {so_was} / {{so_was}} / <TAGS> bleiben unverändert.
    - Juristische Begriffe wie GmbH, AG, UG, OHG, KG, e.K. usw. niemals übersetzen.
    - Wenn source_lang == 'de' oder unbekannt -> Rückgabe = text_src (fail-safe).
    """
    src = (source_lang or "de").lower()
    if src not in SUPPORTED or src == "de" or not text_src:
        return text_src

    client = client or OpenAI()

    system_prompt = (
        "You are a precise translator. Translate into **German** from the given source language.\n"
        f"- Source language (ISO 639-1): {src}\n"
        "- Keep placeholders and variables exactly as-is: {like_this}, {{like_this}}, <TAGS>, $VARS, %(fmt)s.\n"
        "- Do not translate URLs, emails, codes, or content inside {{double braces}}.\n"
        "- Do not translate legal/corporate terms such as GmbH, AG, UG, OHG, KG, e.K., mbH.\n"
        "- Preserve punctuation, line breaks, and Markdown.\n"
        "- Style: concise, polite, clear German."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": text_src}]},
        ],
    )

    return (resp.output_text or "").strip()
