import argparse
from src.bot import chatbot_fn
from gradio import ChatMessage
# Beispiel mit Gradio:
import gradio as gr

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
    return vars(parser.parse_args())

def main(**kwargs):

    share = kwargs.get("share", False)
    debug = kwargs.get("debug", False)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type='messages')
        state = gr.State(value=None)
        txt = gr.Textbox(placeholder="Hier tippen...")

        txt.submit(chatbot_fn, 
                   inputs=[txt, chatbot, state],
                   outputs=[chatbot, state, txt])
        demo.load(lambda: [ChatMessage(
            role='assistant',
            content='Willkommen! Welches Formular möchten Sie ausfüllen?'
        )],
                  outputs=[chatbot])

    demo.launch(debug=debug, share=share)

if __name__ == "__main__":
    args = parse_args()
    main(**args)
