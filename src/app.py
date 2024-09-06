import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base

# List to store publications
publications = []

def add_publication(file, authors, title):
    if file is not None:
        # Save the file
        file_path = os.path.join("uploads", file.name)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)
        
        # Add publication to the list
        publications.append({"authors": authors, "title": title, "file": file.name})
        
        # Return updated list and clear input fields
        return gr.Dataframe.update(value=publications), None, "", ""
    return gr.Dataframe.update(value=publications), None, authors, title

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
                df = pd.DataFrame({}, columns=["authors", "title", "year", "file"])
                publications_list = gr.Dataframe(
                    headers=["authors", "title", "year", "file"],
                    value=df,
                    datatype=["str", "str", "number", "str"],
                    label="Publications List",
                    interactive=False,
                    row_count=10,
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

        add_button.click(
            add_publication,
            inputs=[file_input, authors_input, title_input],
            outputs=[publications_list, file_input, authors_input, title_input]
        )

demo.launch()