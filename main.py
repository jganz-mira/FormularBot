import argparse
from src.bot import chatbot_fn, FORMS
import os
from gradio import ChatMessage
# Beispiel mit Gradio:
import gradio as gr
import uuid
from src.bot_helper import save_responses_to_json
from src.pdf_backend import GenericPdfFiller
from pathlib import Path
from src.translator import translate_from_de, final_msgs, download_button_msgs, files_msgs, pdf_file_msgs

def initial_prompt() -> str:
        """
        Creates an initial prompt which will be uttered by the bot at startup
        """
        prompt = (
            '**Willkommen beim Chatbot der Stadt Göppingen!** Ich werde Sie beim Ausfüllen eines gewünschten Formulars unterstützen. Auf welcher Sprache wollen wir miteinander sprechen?\n'
            '**Welcome to the chatbot for the city of Göppingen!** I will help you fill out the form you need. Which language would you like us to speak?\n'
            '**Bienvenue sur le chatbot de la ville de Göppingen !** Je vais vous aider à remplir le formulaire souhaité. Dans quelle langue voulez-vous que nous parlions ?\n'
            '**Göppingen şehrinin chatbotuna hoş geldiniz!** İstediğiniz formu doldurmanıza yardımcı olacağım. Hangi dilde konuşmak istersiniz?'
        )

        
        # available = sorted(list(FORMS.keys()))
        # # utter available forms to user
        # prompt += "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(available))
        return prompt

def make_and_get_pdf(state: dict) -> str:
    """
    Saves the current conversation state to a JSON file and fills a PDF based on it.
    Returns the path to the generated PDF.
    """
    # Ensure output directory exists
    os.makedirs("out", exist_ok=True)

    # Unique filenames per session
    uid = uuid.uuid4().hex
    json_fname = f"out/{uid}.json"
    pdf_fname = f"out/{uid}.pdf"

    # 1) Save responses to JSON
    save_responses_to_json(state=state, output_path=json_fname)

    # 2) Fill PDF using the JSON
    GenericPdfFiller(json_path=json_fname).fill(output_path=pdf_fname)

    return pdf_fname

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Start the Gradio chatbot with optional sharing.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio's share option")
    
    parser.add_argument("--debug",
        action = "store_true",
        help="Enable debugging output in the terminal")
    
    parser.add_argument(
        "--enable-download",
        action="store_true",
        help="Enable PDF download button in the UI")
    
    return vars(parser.parse_args())

def handle_upload(files, chat, s):
    s = s or {}
    chat = chat or []

    s["uploaded_files"] = files

    if s.get("awaiting_final_upload"):
        # Abschluss: Buttons ausblenden & Schlussnachricht
        s["awaiting_final_upload"] = False
        s["show_upload"] = False     # Upload-Button & Liste ausblenden
        s["completed"] = False       # Download-Button & PDF-Komponente ausblenden

        lang = (s.get("lang") or "de")
        msg = final_msgs.get(lang, final_msgs["de"])

        chat.append(ChatMessage(
            role="assistant",
            content=msg
        ))
        # Immer 3 Outputs: chat, state, files (leer)
        return chat, s, None

    # Normaler Upload mitten im Dialog: Liste anzeigen
    return chat, s, files


def main(**kwargs):

    share = kwargs.get("share", False)
    debug = kwargs.get("debug", False)
    enable_download = kwargs.get("enable_download", False)
    gr.set_static_paths(paths=[Path.cwd().absolute() / "images"])

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type='messages', render_markdown=True)
        state = gr.State(value=None)
        txt = gr.Textbox(placeholder="Hier tippen...")

        # --- UI-Komponenten mit sicheren Defaults erstellen (ohne state.get ...)
        # Button-Text (gr.Button) kommt über value, File-/Files-Label über label
        download_btn = gr.Button(value=download_button_msgs.get('de'), visible=False)
        pdf_file     = gr.File(label=pdf_file_msgs.get('de'), visible=False)

        upload_btn     = gr.UploadButton(label="Upload files", file_count="multiple", visible=False)
        uploaded_files = gr.Files(label=files_msgs.get('de'), visible=False)

        # --- Events
        txt.submit(
            chatbot_fn,
            inputs=[txt, chatbot, state],
            outputs=[chatbot, state, txt]
        )

        demo.load(
            lambda: [ChatMessage(role='assistant', content=initial_prompt())],
            outputs=[chatbot]
        )

        # Download-Bereich dynamisch steuern (Sichtbarkeit + Texte nach Sprache)
        if enable_download:
            state.change(
                lambda s: (
                    # Download-Button
                    gr.update(
                        visible=bool(s and s.get("completed")),
                        value=download_button_msgs.get(
                            (s or {}).get("lang", "de"),
                            download_button_msgs["de"]
                        )
                    ),
                    # PDF-File-Komponente
                    gr.update(
                        visible=bool(s and s.get("completed")),
                        label=pdf_file_msgs.get(
                            (s or {}).get("lang", "de"),
                            pdf_file_msgs["de"]
                        )
                    ),
                ),
                inputs=[state],
                outputs=[download_btn, pdf_file]
            )

            # PDF-Generierung & Download
            download_btn.click(
                make_and_get_pdf,
                inputs=[state],
                outputs=[pdf_file]
            )

        # Upload-Event: Dateien speichern & ggf. Abschlussmeldung + Buttons aus
        upload_btn.upload(
            handle_upload,
            inputs=[upload_btn, chatbot, state],        # files, chat, state
            outputs=[chatbot, state, uploaded_files]    # chat, state, files (Anzeige)
        )

        # Upload-UI (Button + Files-Liste) dynamisch steuern
        state.change(
            lambda s: (
                # UploadButton: Sichtbarkeit + dynamische Beschriftung aus state['upload_label']
                gr.update(
                    visible=bool(s and s.get("show_upload")),
                    label=((s or {}).get("upload_label") or "Dateien hochladen"),
                    value=None  # Niemals Texte als value setzen (value = Dateien)
                ),
                # Files-Liste: Sichtbarkeit, Inhalte, Label nach Sprache
                gr.update(
                    visible=bool(s and s.get("show_upload")),
                    value=((s or {}).get("uploaded_files") if (s and s.get("show_upload")) else None),
                    label=files_msgs.get(
                        (s or {}).get("lang", "de"),
                        files_msgs["de"]
                    )
                )
            ),
            inputs=[state],
            outputs=[upload_btn, uploaded_files]
        )

    demo.launch(debug=debug, share=share)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
