import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base
from metadata_extractor import extract_metadata_new_file
# from tempfile import NamedTemporaryFile
# from llm import LlmBot
from rag_bot import RagBot
import time
from dotenv import load_dotenv
load_dotenv()


# Initialize the LlmBot with the cat persona
bot = RagBot(
    knowledge_base_id=os.environ.get('KNOWLEDGE_BASE_ID'),
    system_prompt="You're a helpful academic who answers questions based on the context provided."
)

def on_submit(message, history):
    return chat(message, history)

def clear_specific_warning(value, warning):
    if value:
        return gr.Markdown(visible=False)
    return warning

def clear_warnings():
    return gr.Markdown(visible=False), gr.Markdown(visible=False), gr.Markdown(visible=False)

def update_metadata_input(file_obj, authors_warning, title_warning, year_warning):
    """Extracts metadata from a PDF file and updates the corresponding input fields

    Args:
        file_obj (gr.File): File object `.pdf`
        authors_warning  (gr.Markdown): Warning if authors metadata is missing
        title_warning  (gr.Markdown): Warning if title metadata is missing
        year_warning (gr.Markdown): Warning if year metadata is missing

    Returns:
    tuple : (str, str, int, gr.Markdown, gr.Markdown, gr.Markdown)
        authors (str): Authors from the metadata. If not found, returns an empty string
        title (str): Title from the metadata. If not found, returns an empty string
        year (int): Year from the metadata. If not found, returns 0
        authors_warning (gr.Markdown): Updated warning for missing authors
        title_warning (gr.Markdown): Updated warning for missing title
        year_warning (gr.Markdown): Updated warning for missing year
    """
    
    if file_obj is not None and file_obj.name.lower().endswith('.pdf'):
        automated_metadata = extract_metadata_new_file(file_obj)
        authors = automated_metadata.get('authors', '')
        title = automated_metadata.get('title', '')
        year = automated_metadata.get('year', 0)
        
        print(automated_metadata)
        
        if authors is None:
            authors = ''
            authors_warning = gr.Markdown('<p style="color: red; font-size: 12px;">⚠️ Authors not found. Please fill in manually.</p>', visible=True)
        
        if title is None:
            title = ''
            title_warning = gr.Markdown('<p style="color: red; font-size: 12px;">⚠️ Title not found. Please fill in manually.</p>', visible=True)
        
        if year is None:
            year = 0
            year_warning = gr.Markdown('<p style="color: red; font-size: 12px;">⚠️ Year not found. Please fill in manually.</p>', visible=True)
        
        return authors, title, year, authors_warning, title_warning, year_warning
    
def upload_file(file_obj, authors, title, year, topic, current_df):
    """Upload a file to S3 and resync the Bedrock knowledge base
    Args:
        file_obj (File): File object to upload
        authors (str): Authors of the publication
        title (str): Title of the publication
        year (int): Year of the publication
        topic (str): Topic of the publication
        current_df (pd.DataFrame): Current DataFrame of publications
    Returns:
        pd.DataFrame: Updated DataFrame of publications
        str: Authors of the publication
        str: Title of the publication
        int: Year of the publication
        str: Topic of the publication
        File: File object
    """
    metadata = {
        'authors': authors,
        'title': title,
        'year': str(year),
        'topic': topic
    }
    
    try:
        upload_file_to_s3(file_path=file_obj.name, metadata=metadata)
        resync_bedrock_knowledge_base()
        upload_success = True
    except Exception as e:
        print(f"Error uploading file: {e}")
        upload_success = False
    if upload_success:
        metadata["file"] = os.path.basename(file_obj.name)
        new_row = pd.DataFrame([metadata])
        new_row = new_row.drop(columns=["topic"])
        new_row = format_metadata(new_row)
        updated_df = pd.concat([new_row, current_df], ignore_index=True).reset_index(drop=True)
        return updated_df, "", "", "", "", None, gr.Button(
            value="Upload successful!", interactive=False
        )
    else:
        return current_df, authors, title, year, topic, file_obj, gr.Button("Submit")


def format_metadata(df, max_length=50):
    """Format metadata for display in the Gradio Dataframe
    Args:
        df (pd.DataFrame): DataFrame containing metadata
        max_length (int): Maximum length for titles and authors
    Returns:
        pd.DataFrame: DataFrame with shortened titles and authors    
    """
    longer_titles = df['title'].str.len() > max_length
    longer_authors = df['authors'].str.len() > max_length
    df.loc[longer_titles, 'title'] = df.loc[longer_titles, 'title'].str.slice(0, max_length) + '...'
    df.loc[longer_authors, 'authors'] = df.loc[longer_authors, 'authors'].str.slice(0, max_length) + '...'
    # get rid of file column
    df = df.drop(columns=["file"])
    # drop empty rows
    df = df.dropna()
    return df


def reset_button():
    """Reset the button after 2 seconds"""
    time.sleep(2)  # Wait for 2 seconds
    return gr.Button(value="Submit", interactive=True)

def chat(message, history):
    bot_response = bot.chat(message)
    history.append((message, bot_response))
    return "", history

def update_selected(n, academic, educator, fun_mode, custom, custom_prompt_text):
    """Update the selected behavior and system prompt
    Args:
        n (int): Selected behavior
        academic (Button): Academic behavior button
        educator (Button): Educator behavior button
        fun_mode (Button): Fun mode behavior button
        custom (Button): Custom behavior button
        custom_prompt_text (str): Custom system prompt text    
    Returns:
        list[Button]: Updated behavior buttons
        Textbox: Custom system prompt textbox
    """

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

# Filter functionality functions

def change_filter_years(start, end):
    # This is a dummy function for now
    print(f"Filtering years: start={start}, end={end}")
    bot.retriever.filter_years(start, end)

def toggle_year_filter(checked):
    return gr.update(visible=checked)

def handle_year_filter(checked, from_year, to_year):
    if checked:
        start = int(from_year) if from_year.strip() else 1800
        end = int(to_year) if to_year.strip() else 2100
        if start > end:
            end = 2100
    else:
        start, end = 1800, 2100
    
    change_filter_years(start, end)


# topic stuff
def toggle_topic_filter(checked):
    return gr.update(visible=checked)

def change_filter_topic(topic):
    print(f"Filtering topic: {topic}")
    bot.retriever.filter_topic(topic)

def handle_topic_filter(checked, topic):
    if checked:
        if topic is not None:
            topic = str(topic) if topic.strip() else None
        change_filter_topic(topic)
    else:
        change_filter_topic(None)

# Gradio interface
with gr.Blocks(fill_width=True) as demo:
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
                with gr.Group():
                    with gr.Row():
                        # Checkbox for filtering by Year
                        with gr.Column(scale=1):
                            filter_by_year = gr.Checkbox(label="Filter by Year", value=False)
                            with gr.Row(visible=False) as year_filter:
                                year_from = gr.Textbox(label="From", lines=1)
                                year_to = gr.Textbox(label="To", lines=1)
                        # Checkbox for filtering by Topic
                        with gr.Column(scale=1):
                            filter_by_topic = gr.Checkbox(label="Filter by Topic", value=False)
                            with gr.Row(visible=False) as topic_filter:
                                topic_dropdown = gr.Dropdown(choices=["Machine learning", "Biology", "UQ"], label="Topic")
                        
                # filter_by_year.change(
                #     fn=toggle_year_filter, 
                #     inputs=filter_by_year, 
                #     outputs=year_filter
                # )

                # # Add new event listener for year filter changes
                # filter_by_year.change(
                #     fn=handle_year_filter,
                #     inputs=[filter_by_year, year_from, year_to]
                # )
                # year_from.change(
                #     fn=handle_year_filter,
                #     inputs=[filter_by_year, year_from, year_to]
                # )
                
                # year_to.change(
                #     fn=handle_year_filter,
                #     inputs=[filter_by_year, year_from, year_to]
                # )
                
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

                submit.click(on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
                msg.submit(on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
                clear.click(lambda: None, None, chatbot, queue=False)

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

            with gr.Tab("Publications"):
                with gr.Column():
                    with gr.Row(70):
                        df = get_s3_metadata()
                        # shorten long titles and authors and add ... at the end when shortened
                        df = format_metadata(df)
                        # make the theme text size smaller
                        publications_list = gr.Dataframe(
                            headers=["authors", "title", "year"],
                            value=df,
                            datatype=["str", "str", "number"],
                            label="Publications List",
                            interactive=False,
                            row_count=8,
                            line_breaks=False,
                            column_widths=[20, 20, 3],
                            height=300
                        )
                    with gr.Row(30, variant="compact"):
                        # Left 80% for text inputs
                        with gr.Column(scale=70):
                            authors_input = gr.Textbox(label="Authors")
                            title_input = gr.Textbox(label="Title")
                            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])

                        # Right 20% for file upload
                        with gr.Column(scale=30, variant="compact"):
                            year_input = gr.Number(label="Year")
                            topic_input = gr.Textbox(label="Topic")
                            add_button = gr.Button("Add Publication")
                
                authors_warning = gr.Markdown("", visible=False)
                title_warning = gr.Markdown("", visible=False)
                year_warning = gr.Markdown("", visible=False)
                
                # Add event listener for file upload
                file_input.upload(
                    fn=update_metadata_input,
                    inputs=[file_input, authors_warning, title_warning, year_warning],
                    outputs=[authors_input, title_input, year_input, authors_warning, title_warning, year_warning]
                )
                
                # Add event listener for file removal
                file_input.clear(
                    fn=clear_warnings,
                    inputs=[],
                    outputs=[authors_warning, title_warning, year_warning]
                )
                
                # Add event listeners for manual input
                authors_input.change(
                    fn=clear_specific_warning,
                    inputs=[authors_input, authors_warning],
                    outputs=[authors_warning]
                )
                title_input.change(
                    fn=clear_specific_warning,
                    inputs=[title_input, title_warning],
                    outputs=[title_warning]
                )
                year_input.change(
                    fn=clear_specific_warning,
                    inputs=[year_input, year_warning],
                    outputs=[year_warning]
                )
                
                add_button.click(
                    fn=upload_file, 
                    inputs=[
                        file_input, authors_input, title_input, year_input, topic_input, publications_list
                    ],
                    outputs=[
                        publications_list, authors_input, title_input, year_input, topic_input, file_input,
                        add_button
                    ]
                ).then(
                    fn=reset_button,
                    outputs=add_button
                )

                
                filter_by_topic.change(
                    fn=toggle_topic_filter, 
                    inputs=filter_by_topic, 
                    outputs=topic_filter
                )
                
                filter_by_year.change(
                    fn=toggle_year_filter, 
                    inputs=filter_by_year, 
                    outputs=year_filter
                )

                # Add new event listener for year filter changes
                filter_by_year.change(
                    fn=handle_year_filter,
                    inputs=[filter_by_year, year_from, year_to]
                )
                year_from.change(
                    fn=handle_year_filter,
                    inputs=[filter_by_year, year_from, year_to]
                )
                
                year_to.change(
                    fn=handle_year_filter,
                    inputs=[filter_by_year, year_from, year_to]
                )
                
                # add event listener for topic filter changes
                filter_by_topic.change(
                    fn=handle_topic_filter,
                    inputs=[filter_by_topic, topic_dropdown]
                )
                
                topic_dropdown.change(
                    fn=handle_topic_filter,
                    inputs=[filter_by_topic, topic_dropdown]
                )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get('SERVER_IP'),
        server_port=int(os.environ.get('SERVER_PORT')),
        debug=True
    )

