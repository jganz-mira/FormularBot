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

def main(**kwargs):

    share = kwargs.get("share", False)
    debug = kwargs.get("debug", False)
    enable_download = kwargs.get("enable_download", False)
    gr.set_static_paths(paths=[Path.cwd().absolute() / "images"])

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type='messages', render_markdown=True)
        state = gr.State(value=None)
        txt = gr.Textbox(placeholder="Hier tippen...")

        download_btn = gr.Button("PDF erzeugen & herunterladen", visible=False)
        pdf_file     = gr.File(label="Dein ausgefülltes Formular", visible=False)

        txt.submit(chatbot_fn, 
                   inputs=[txt, chatbot, state],
                   outputs=[chatbot, state, txt])
        demo.load(lambda: [ChatMessage(
            role='assistant',
            content=initial_prompt()
        )],
                  outputs=[chatbot])
        
        if enable_download:
            state.change(
                lambda s: (
                    gr.update(visible=bool(s.get("completed"))),
                    gr.update(visible=bool(s.get("completed")))
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

    demo.launch(debug=debug, share=share)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
