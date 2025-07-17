# Stadt Göppingen – Chatbot & PDF Filler

Welcome to the repository for the City of Göppingen's chatbot project. The bot guides citizens through administrative forms via dialog, validates their input, and fills in the appropriate PDF document.

---

## Contents

1. [Start the Server](#start-the-server)
2. [Add New Forms](#add-new-forms)
3. [Form JSON Structure](#form-json-structure)
4. [Project Structure & Architecture](#project-structure--architecture)
5. [License](#license)

---

## Start the Server

### 1. Requirements



### 2. Installation


### 3. Environment Variables (optional)

- `LLM_ENDPOINT` – URL of a local LLM validator (default: `http://localhost:8080/completion`)
- `OPENAI_API_KEY` – automatically read from `.key`, can be overridden

### 4. Start the Bot

```bash
# Default (local) mode
$ python src/main.py

# With public Gradio share link & debug output
$ python src/main.py --share --debug

# Enable download button for the filled PDF
$ python src/main.py --enable-download
```

On startup, the bot displays a selection of all JSON forms in the `forms/ge` folder.

---

## Add New Forms

1. Place a **PDF template** *(with internal form fields)* in the `pdfs/` folder, e.g. `Gewerbeanmeldung.pdf`.
2. Create the **JSON definition** `forms/ge/<Name>.json` – see [Structure](#form-json-structure).\
   *The filename (without **`.json`**) serves as the form identifier shown to users.*
3. **Validators (optional):**
   - Use methods from `BaseValidators` for simple fields.
   - For new forms, you can create a validators class in `src/validators.py` by inhereting from BaseValidators (e.g. `GewerbeanmeldungValidators`).
   - Register the class in the `validator_map` in `src/bot.py`.
4. **Restart the server** – the form will appear automatically in the bot selection.

> **Tip:** Wehn implementing a new form, start with a minimal demo file (`Gewerbeanmeldung_demo.json`) and expand it step by step.

---

## Form JSON Structure

```jsonc
{
  "slots": [ /* 1. Slots */ ],
  "prompt_map": { /* 2. Prompts */ },
  "validators": "GewerbeanmeldungValidators",  // 3. Validator class
  "pdf_file": "pdfs/Gewerbeanmeldung.pdf"      // 4. PDF template
}
```

### 1. `slots` – Questions & Data Fields

Each object represents **one question**, and also defines **target field mapping** for the PDF.

| Key                   | Type / Example                                          | Description                                |
| --------------------- | ------------------------------------------------------- | ------------------------------------------ |
| `slot_name`           | "family\_name"                                          | Internal name (must be unique)             |
| `slot_type`           | `"text"` \| `"choice"`                                  | Input type                                 |
| `description`         | "Family name; last name"                                | Description for change intent detection    |
| `filed_name`          | "txtFamiliennameS1" or `["chkJa", "chkNein"]`           | PDF form field name(s) / checkboxes        |
| `choices`             | `["yes", "no"]`                                         | List of options for `choice` slots         |
| `condition`           | `{ "slot_name": "nationality", "slot_value": "other" }` | Show question only if condition is met     |
| `check_box_condition` | "false"                                                 | Special mapping bool → checkbox (see code) |

#### Conditions (`condition`)

- **Equals:** `slot_value: "yes"`
- **List:** `slot_value: ["Opt A", "Opt B"]` → Only shown if previous answer is in the list.
- **Not empty:** `slot_value: "not empty"` → Shown as soon as referenced field is filled.

### 2. `prompt_map`

Language-dependent prompts shown by the bot – structured as `{ <lang-code>: { <slot_name>: <prompt> } }`.\
Currently only **German (**``**)** is used, but multilingual support is built-in.\


### 3. `validators`

Name of the Python class in `src/validators.py` which provides methods like `valid_<slot_name>()`.\
If no method is defined, user input will be accepted without checking.

### 4. `pdf_file`

Relative path to the PDF template. For each new form, a template with readable filed names must be present.\
When filling in, `GenericPdfFiller` uses slot mappings for text & checkboxes to populate the form.

---

## Project Structure & Architecture

```
├─ src/
│  ├─ bot.py                 # Dialog logic & Gradio callback
│  ├─ bot_helper.py          # Helper functions (slot selection, JSON export, …)
│  ├─ validators.py          # Validation classes (LLM-based & regex-based)
│  ├─ pdf_backend.py         # PDF form filling via PyPDF & pikepdf
│  ├─ llm_validator_service.py# Wrapper for local/remote LLM requests
│  └─ main.py                # Gradio UI & CLI entry point
├─ forms/
│  └─ ge/                    # All form definitions (*.json)
├─ pdfs/                     # PDF templates
└─ out/                      # Generated JSON/PDF at runtime
```

> The core function `chatbot_fn` processes each user message, determines the next slot (`next_slot_index`), and validates the input before storing it in `state["responses"]`. **It is called each time a user sends a promt to the bot**.

---
