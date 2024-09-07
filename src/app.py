import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base
from tempfile import NamedTemporaryFile
from llm import LlmBot
from rag_bot import RagBot

# PLACEHOLDER FOR THE API KEYS
import os
# os.environ['AWS_ACCESS_KEY_ID'] = ''
# os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'
# os.environ['AWS_SESSION_TOKEN'] = ''

bot = RagBot(knowledge_base_id="4FYUGYITNF", system_prompt="You're a helpful academic knowledge machine who answers questions based on the context provided.")

def chat(message, history):
    bot_response = bot.chat(message)
    history.append((message, bot_response))
    return "", history

with gr.Blocks() as demo:
    with gr.Row():
        # Left column for behavior selection
        with gr.Column(scale=1):
            gr.Markdown("### Chatbot Behavior")
            with gr.Column():
                academic_btn = gr.Button("Academic", variant="primary")
                educator_btn = gr.Button("Educator", variant="secondary")
                fun_mode_btn = gr.Button("Fun Mode", variant="secondary")
                custom_btn = gr.Button("Custom", variant="secondary")
                custom_prompt = gr.Textbox(label="Custom System Prompt", placeholder="Enter your custom prompt here", visible=False)

        # Right column for chat and publications
        with gr.Column(scale=4):
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot()
                msg = gr.Textbox(lines=1, label="Message")
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

            with gr.Tab("Publications"):
                with gr.Column():
                    # Publications list (top 75%)
                    with gr.Row():
                        df = get_s3_metadata()
                        # shorten long titles and authors and add ... at the end when shortened
                        max_length = 60
                        longer_titles = df['title'].str.len() > max_length
                        longer_authors = df['authors'].str.len() > max_length
                        df.loc[longer_titles, 'title'] = df.loc[longer_titles, 'title'].str.slice(0, max_length) + '...'
                        df.loc[longer_authors, 'authors'] = df.loc[longer_authors, 'authors'].str.slice(0, max_length) + '...'
                        publications_list = gr.Dataframe(
                            headers=["authors", "title", "year", "file"],
                            value=df,
                            datatype=["str", "str", "number", "str"],
                            label="Publications List",
                            interactive=False,
                            row_count=8,
                            line_breaks=True,
                            column_widths=[22, 22, 5, 8]
                        )
                    
                    # Input area (bottom 25%)
                    with gr.Row():
                        # Left 80% for text inputs
                        with gr.Column(scale=20):
                            authors_input = gr.Textbox(label="Authors")
                            title_input = gr.Textbox(label="Title")
                            year_input = gr.Number(label="Year")
                            add_button = gr.Button("Add Publication")
                        
                        # Right 20% for file upload
                        with gr.Column(scale=20):
                            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])

    def on_submit(message, history):
        return chat(message, history)

    submit.click(on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    def update_selected(n, academic, educator, fun_mode, custom, custom_prompt_text):
        buttons = [academic, educator, fun_mode, custom]
        if n == academic:
            prompt = "You're a helpful academic who answers questions based on the context provided."
            custom_prompt_visible = False
        elif n == educator:
            prompt = "You're a helpful educator who explains concepts clearly and patiently."
            custom_prompt_visible = False
        elif n == fun_mode:
            prompt = "Pretend you're a helpful, talking cat. Meow!"
            custom_prompt_visible = False
        else:  # custom
            prompt = custom_prompt_text if custom_prompt_text else "Enter your custom prompt above."
            custom_prompt_visible = True
        
        bot.llm.change_system_prompt(prompt)
    
        return [gr.update(variant="primary" if i == n else "secondary") for i in buttons] + [gr.update(visible=custom_prompt_visible)]

    # Connect behavior selection buttons
    academic_btn.click(update_selected, 
                       inputs=[academic_btn, academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt],
                       outputs=[academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt])
    educator_btn.click(update_selected, 
                       inputs=[educator_btn, academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt],
                       outputs=[academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt])
    fun_mode_btn.click(update_selected, 
                       inputs=[fun_mode_btn, academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt],
                       outputs=[academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt])
    custom_btn.click(update_selected, 
                     inputs=[custom_btn, academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt],
                     outputs=[academic_btn, educator_btn, fun_mode_btn, custom_btn, custom_prompt])

    # Add event for custom prompt changes
    custom_prompt.change(
        lambda prompt: bot.llm.change_system_prompt(prompt),
        inputs=[custom_prompt],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=2233, debug=True)