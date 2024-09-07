import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base
from tempfile import NamedTemporaryFile

# PLACEHOLDER FOR THE API KEYS

#not sure it really works
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Varela+Round&display=swap');

body {
    font-family: 'Varela Round', sans-serif;
}
"""

custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    # font=("Roboto", "sans-serif"),
).set(
    body_background_fill="#f5e6d3",  # Light orange-brown
    body_text_color="#4a3728",  # Dark brown for text
    color_accent_soft="#b3470c",  # Soft dark orange (matching button color)
    background_fill_primary="#faf0e6",  # Very light orange-brown
    background_fill_secondary="#f8e5d3",  # Slightly darker orange-brown
    border_color_primary="#e6ccb2",  # Light brown border
    button_primary_background_fill="#b3470c",  # Specified dark orange for buttons
    button_primary_background_fill_hover="#b54404",  # Lighter version for hover
    button_primary_text_color="white",
    button_secondary_background_fill="#b3470c",  # Same color for secondary buttons
    button_secondary_background_fill_hover="#b54404",  # Same hover color for secondary buttons
    button_secondary_text_color="white",
    input_background_fill="#fff5e6",  # Very light orange for input fields
    input_border_color="#e6ccb2",  # Light brown border for inputs
    input_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
    block_background_fill="#fff5e6",  # Very light orange for blocks
)

def chat(message, history):
    # Placeholder for actual chatbot logic
    bot_message = f"You said: {message}"
    history.append((message, bot_message))
    return "", history

def handle_upload(file_input, title, authors, year):
    metadata = {
        'title': title,
        'authors': authors,
        'year': year
    }
    upload_file_to_s3(file_input, metadata)
    return None

def refresh_publications():
    df = get_s3_metadata()
    # Shorten long titles and authors and add '...' at the end when shortened
    max_length = 60
    df['title'] = df['title'].apply(lambda x: x if len(x) <= max_length else x[:max_length] + '...')
    df['authors'] = df['authors'].apply(lambda x: x if len(x) <= max_length else x[:max_length] + '...')
    return df

with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
    
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
                    headers=["Authors", "Title", "Year", "File"],
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
        
        #how to enforce these?
        custom_css = """
            #left_column, #right_column {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }

        #left_column .gr-box {
            margin-top: auto;
            margin-bottom: auto;ß
            text-align: center;
        }
        """
                
        def add_publication(file_input, title_input, authors_input, year_input):
            handle_upload(file_input, title_input, authors_input, year_input)
            # Clear the input fields
            return gr.update(value=refresh_publications())
        
        def clear_inputs():
            return [None, "", "", None]  # Clear file_input, title_input, authors_input, year_input
        
        # when add_button is clicked and file is uploaded, upload file to S3
        add_button.click(
            add_publication, 
            [file_input, title_input, authors_input, year_input], 
            [publications_list]
        ).then(
            clear_inputs,  # This function will be called after add_publication
            outputs=[file_input, title_input, authors_input, year_input]
        )
                
demo.launch()