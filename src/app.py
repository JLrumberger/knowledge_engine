import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base
from tempfile import NamedTemporaryFile

# PLACEHOLDER FOR THE API KEYS

def chat(message, history):
    # Placeholder for actual chatbot logic
    bot_message = f"You said: {message}"
    history.append((message, bot_message))
    return "", history


with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(lines=3)
        clear = gr.Button("Clear")

        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

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
        metadata = {
            'title': title_input,'authors': authors_input, 'year': year_input
        }
        # when add_button is clicked and file is uploaded, upload file to S3
        #add_button.click(upload_file_to_s3, [file_input, metadata], [file_input, metadata])

demo.launch()