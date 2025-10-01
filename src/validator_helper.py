import json

def response_to_dict(resp):
    """
    Wandelt eine OpenAI Responses-API Antwort in ein dict um.
    Unterstützt Structured Outputs (output_parsed) und JSON-Mode (output_text).
    """
    # 1) Structured Outputs: direkt geparst vom SDK
    if hasattr(resp, "output_parsed") and resp.output_parsed is not None:
        return resp.output_parsed.model_dump()

    # 2) JSON-Mode: Text in JSON umwandeln
    raw = getattr(resp, "output_text", None)
    if not raw:
        try:
            raw = resp.output[0].content[0].text
        except Exception:
            raise ValueError("Keine JSON-Ausgabe gefunden")

    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").split("\n", 1)[-1]

    return json.loads(clean)


def convert_to_bool(input):

    if input == 'VALID':
        return True
    elif input == 'INVALID':
        return False
    else:
        raise f"Invalid Input {input} for conversion, must either be 'VALID' or 'INVALID' not {input}"
    
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def is_gp_town(postal_code):
    if postal_code not in [73033, 73035, 73037, 73116]:
        return "\n\n**Hinweis:** Es scheint, als wollten Sie ein Gewerbe anmelden, dessen Betriebsstätte nicht im Zuständigkeitsbereich des Gewerbeamts Göppingen liegt. Bitte wenden Sie sich in diesem Fall an das für Sie zuständige Gewerbeamt oder geben Sie die korrekte Adresse ein.\n\n"
    else:
        return ""