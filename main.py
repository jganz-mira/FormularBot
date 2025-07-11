from src.bot import chatbot_fn
from gradio import ChatMessage
# Beispiel mit Gradio:
import gradio as gr

def main():
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

    demo.launch(debug=True)

if __name__ == "__main__":
    main()
