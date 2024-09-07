import gradio as gr
import pandas as pd
import os
from aws_helpers import upload_file_to_s3, get_s3_metadata, resync_bedrock_knowledge_base
from tempfile import NamedTemporaryFile
from llm import LlmBot
from rag_bot import RagBot
import time
from dotenv import load_dotenv
load_dotenv()


# Initialize the LlmBot with the cat persona
bot = RagBot(
    knowledge_base_id=os.environ.get('KNOWLEDGE_BASE_ID'),
    system_prompt="Pretend you're a helpful, talking cat. Meow!"
)

def on_submit(message, history):
    return chat(message, history)


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


def format_metadata(df, max_length=70):
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
                    with gr.Row(70,variant="compact"):
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
                            column_widths=[22, 22, 5],
                        )
                    with gr.Row(30, variant="compact"):
                        # Left 80% for text inputs
                        with gr.Column(scale=70, variant="compact"):
                            authors_input = gr.Textbox(label="Authors")
                            title_input = gr.Textbox(label="Title")
                            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])

                        # Right 20% for file upload
                        with gr.Column(scale=30, variant="compact"):
                            year_input = gr.Number(label="Year")
                            topic_input = gr.Textbox(label="Topic")
                            add_button = gr.Button("Add Publication")

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


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get('SERVER_IP'),
        server_port=int(os.environ.get('SERVER_PORT')),
        debug=True
    )

