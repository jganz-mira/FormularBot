import json
import os
from PyPDF2 import PdfReader, PdfWriter

class GenericPdfFiller:
    """
    A universal PDF filler that reads a JSON payload with form data and target field mappings,
    handles text concatenation for shared fields, and checkbox/radio button logic based on choices.
    """
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.template = payload["pdf_file"]
        self.responses = payload["data"]

    def fill(self, output_path: str):
        # 1. Build field_map: field_name -> value or checkbox status
        field_map = {}
        text_accum = {}

        for slot, details in self.responses.items():
            value = details.get("value")
            targets = details.get("target_filed_name")
            choices = details.get("choices")

            if choices:
                # Handle choice/boolean mapping
                for idx, field_name in enumerate(targets):
                    # Determine selection
                    is_selected = False
                    if isinstance(value, str) and value.lower() in ("true", "false"):
                        val_bool = value.lower() in ("true", "ja", "yes", "1", "on")
                        is_selected = (val_bool and idx == 0) or (not val_bool and idx == 1)
                    else:
                        is_selected = str(value).strip().lower() == str(choices[idx]).lower()
                    if is_selected:
                        field_map[field_name] = "/Y"
            else:
                # Accumulate text fields
                if isinstance(targets, list):
                    for field_name in targets:
                        text_accum.setdefault(field_name, []).append(value)
                else:
                    text_accum.setdefault(targets, []).append(value)

        # 2. Flatten text accumulations
        for field_name, vals in text_accum.items():
            non_empty = [v for v in vals if v]
            field_map[field_name] = ", ".join(non_empty)

        # 3. Read, fill, and write PDF
        reader = PdfReader(self.template)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        
        for i in range(len(writer.pages)):
            writer.update_page_form_field_values(writer.pages[i], field_map)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)
        
        # for debugging
        # return field_map