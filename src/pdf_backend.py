import os
import json

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, BooleanObject
import pikepdf


class GenericPdfFiller:
    """
    A universal PDF filler that reads a JSON payload with form data and target field mappings,
    handles text concatenation for shared fields, and checkbox/radio button logic based on choices.
    After filling, sets NeedAppearances and makes widget backgrounds transparent.
    """
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.template = payload["pdf_file"]
        self.responses = payload["data"]

    def fill(self, output_path: str):
        # 1. Feld-Map bauen
        field_map = {}
        text_accum = {}
        for slot, details in self.responses.items():
            value   = details.get("value")
            targets = details.get("target_filed_name")
            choices = details.get("choices")
            check_box_condition = details.get("check_box_condition")
            if choices:
                for idx, fn in enumerate(targets):
                    # Wahrheits-Abgleich
                    if isinstance(value, str) and value.lower() in ("true","false"):
                        if check_box_condition is not None:
                            selected = value.lower() == check_box_condition
                        else:
                            val_bool = value.lower() in ("true","ja","yes","1","on")
                            selected = (val_bool and idx==0) or (not val_bool and idx==1)
                    else:
                        selected = str(value).strip().lower() == str(choices[idx]).lower()
                    if selected:
                        field_map[fn] = "/Y"
            else:
                # Text sammeln
                if isinstance(targets, list):
                    for fn in targets:
                        text_accum.setdefault(fn,[]).append(value)
                else:
                    text_accum.setdefault(targets,[]).append(value)

        # 2. Text flachlegen
        for fn, vals in text_accum.items():
            non_empty = [v for v in vals if v]
            field_map[fn] = ", ".join(non_empty)

        # 3. PDF laden & Felder füllen
        reader = PdfReader(self.template)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        # 3a) AcroForm-Objekt übernehmen
        acro = reader.trailer["/Root"].get("/AcroForm")
        if acro is not None:
            writer._root_object[NameObject("/AcroForm")] = acro

        # 3b) Feldwerte schreiben
        for pg in writer.pages:
            writer.update_page_form_field_values(pg, field_map)

        # 4. NeedAppearances setzen
        root = writer._root_object
        acro = root.get("/AcroForm")
        if acro is None:
            acro = writer._add_object({})
            root[NameObject("/AcroForm")] = acro
        acro.update({ NameObject("/NeedAppearances"): BooleanObject(True) })

        # 5. Zwischenspeichern
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)

        # 6. Post-Processing mit pikepdf:
        #    a) Widget-Hintergründe entfernen
        #    b) Checkboxen setzten (Annotation /AS + /V auf den „On“-State)
        with pikepdf.open(output_path, allow_overwriting_input=True) as pdf:
            for page in pdf.pages:
                # statt page.Annots oder [] jetzt:
                for annot in page.get("/Annots", []):
                    if annot.get("/Subtype") == pikepdf.Name("/Widget"):
                        # a) Hintergrund & Rahmenfarbe löschen
                        mk = annot.get("/MK")
                        if mk is not None:
                            if "/BG" in mk:    del mk["/BG"]
                            if "/BC" in mk:    del mk["/BC"]

                        # b) Checkbox ankreuzen
                        field_name = annot.get("/T")
                        if field_name and field_map.get(field_name) == "/Y":
                            ap_dict = annot.get("/AP", {}).get("/N", {})
                            for state in ap_dict.keys():
                                if state != "/Off":
                                    annot["/AS"] = pikepdf.Name(state)
                                    annot["/V"]  = pikepdf.Name(state)
                                    break

            pdf.save(output_path)
