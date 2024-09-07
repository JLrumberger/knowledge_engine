import boto3
import pandas as pd
import time
import logging
import json
import os
import re
from pypdf import PdfReader
from llm import LlmBot
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def upload_file_to_s3(
        file_path, metadata, bucket_name=os.environ.get('BUCKET_NAME'), object_name=None,
    ):
    """Upload a file to an S3 bucket with metadata
    Args:
        file_path (str): Local path of the file to upload
        metadata (dict): Metadata to be stored with the file
            {
                'title': 'Dropout as a bayesian approximation',
                'authors': 'Gal, Yarin and Ghahramani, Zoubin',
                'year': 2016,
                'topic': 'Machine Learning
            }
        bucket_name (str): Name of the bucket to upload to
        object_name (str): S3 object name. If not specified then file_name is used
    """
    s3 = boto3.client('s3')
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    # Ensure metadata is a dictionary
    if metadata is None:
        metadata = {}
    
    # Prepare the ExtraArgs parameter
    extra_args = {
        'Metadata': metadata,
        'ContentType': 'application/pdf'  # Adjust this based on your file type
    }
    
    try:
        s3.upload_file(file_path, bucket_name, object_name, ExtraArgs=extra_args)
        logger.info(
            f"File {file_path} uploaded successfully to {bucket_name}/{object_name} with metadata"
        )
    except Exception as e:
        logger.info(f"Error uploading file: {e}")
    # upload metadata file to s3
    metadata_file = {
    "metadataAttributes": {
        "name": metadata["title"],
        "year": metadata["year"], 
        "type": metadata["topic"],
        }
    }
    # turn metadata file into json
    metadata_file = json.dumps(metadata_file)
    # upload metadata file to s3
    s3.put_object(Bucket=bucket_name, Key=f"{object_name}.metadata.json", Body=metadata_file)


def get_s3_metadata(bucket_name=os.environ.get('BUCKET_NAME')):
    """Get metadata for all objects in an S3 bucket
    Args:
        bucket_name (str): Name of the bucket to list objects from
    Returns:
        pd.DataFrame: DataFrame containing metadata for each object in the bucket
    """
    s3 = boto3.client('s3')
    metadata_list = []

    try:
        # List all objects in the bucket
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                # Get metadata for each object
                response = s3.head_object(Bucket=bucket_name, Key=obj['Key'])
                
                # Combine object info with metadata
                metadata = {
                    'Key': obj['Key'],
                    'Size': obj['Size'],
                    'LastModified': obj['LastModified'],
                    **response.get('Metadata', {}),
                    'ContentType': response.get('ContentType'),
                    'ETag': response.get('ETag'),
                }
                metadata_list.append(metadata)

    except Exception as e:
        logger.info(f"Error: {e}")
        return None

    # Create DataFrame from the list of metadata dictionaries
    df = pd.DataFrame(metadata_list)
    df = df[["authors", "title", "year", "Key"]].rename(columns={"Key": "file"})
    return df


def resync_bedrock_knowledge_base(
        knowledge_base_id=os.environ.get('KNOWLEDGE_BASE_ID'),
        data_source_id=os.environ.get('DATA_SOURCE_ID'), wait_for_completion=False
    ):
    """Re-sync a Bedrock knowledge base with an S3 bucket
    Args:
        knowledge_base_id (str): ID of the knowledge base to re-sync
        data_source_id (str): ID of the data source to re-sync
        wait_for_completion (bool): Whether to wait for the re-sync job to complete
    """
    bedrock = boto3.client('bedrock-agent', region_name=os.environ.get('AWS_DEFAULT_REGION'))
    try:
        # Start a new ingestion job
        response = bedrock.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            description='Re-sync knowledge base with S3 bucket'
        )
        
        job_id = response['ingestionJob']['ingestionJobId']
        logger.info(f"Ingestion job started. Job ID: {job_id}")
        
        # Wait for the job to complete
        if wait_for_completion:
            while True:
                job_status = bedrock.get_ingestion_job(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                    ingestionJobId=job_id
                )['ingestionJob']['status']
                
                if job_status == 'COMPLETE':
                    logger.info("Re-sync completed successfully.")
                    break
                elif job_status in ['FAILED', 'CANCELLED']:
                    logger.info(f"Re-sync failed. Status: {job_status}")
                    break
                else:
                    logger.info(f"Re-sync in progress. Current status: {job_status}")
                    time.sleep(1)  # Wait for 30 seconds before checking again
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")


def invoke_agent_helper(
        query, session_id, agent_id, alias_id, enable_trace=False, session_state=None
    ):
    """Helper function to invoke a Bedrock agent with the given query
    Args:
        query (str): Input query to the agent
        session_id (str): Unique session ID for the conversation
        agent_id (str): ID of the agent to invoke
        alias_id (str): ID of the agent alias to use
        enable_trace (bool): Whether to enable tracing for the agent invocation
        session_state (dict): State information from the previous session
    Returns:
        str: Response from the agent
    """
    bedrock_agent_runtime_client = boto3.client(
        'bedrock-agent-runtime', region_name=os.environ.get('AWS_DEFAULT_REGION')
    )
    end_session: bool = False
    if not session_state:
        session_state = {}

    # invoke the agent API
    agent_response = bedrock_agent_runtime_client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=enable_trace,
        endSession=end_session,
        sessionState=session_state
    )

    if enable_trace:
        logger.info(agent_response)

    event_stream = agent_response['completion']
    try:
        for event in event_stream:
            if 'chunk' in event:
                data = event['chunk']['bytes']
                if enable_trace:
                    logger.info(f"Final answer ->\n{data.decode('utf8')}")
                agent_answer = data.decode('utf8')
                return agent_answer
                # End event indicates that the request finished successfully
            elif 'trace' in event:
                if enable_trace:
                    logger.info(json.dumps(event['trace'], indent=2))
            else:
                raise Exception("unexpected event.", event)
    except Exception as e:
        raise Exception("unexpected event.", e)

def initialize_bot(text: str):
    prompt = "Pretend to be a helpful research assistant"
    query = f"What are the authors' names and title contained in this first page of paper: {text}? \n" \
        "And do you know the publication year? \n" \
        "In addition, you can always respond in the same format:\n" \
        "Authors: [...] \n" \
        "Title: [...] \n" \
        "The publication year [...]. The submission happened on [...] "
    bot = LlmBot(system_prompt=prompt)
    return bot, query

def read_pdf(paper_path: str):
    reader = PdfReader(paper_path)
    first_page = reader.pages[0] #read first page
    text = first_page.extract_text()   
    return text

def preprocess_info(response: str):
    authors_match = re.search(r"Authors:\s*(.*?)(?=\s*Title:|\s*The publication year)", response, re.DOTALL)
    if authors_match:
        authors_raw = authors_match.group(1).replace("*", "").replace("\n", "").strip()
        authors_list = [author.strip() for author in re.split(r',\s*|\sand\s', authors_raw)]
        reordered_authors = []
        for author in authors_list:
            name_parts = author.split()
            if len(name_parts) > 1:
                reordered_author = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
                reordered_authors.append(reordered_author)
            else:
                reordered_authors.append(author)
        authors = ' and '.join(reordered_authors)
    else:
        authors = None

    title_match = re.search(r"Title:\s*(.*?)(?=\s*The publication year|\s*The paper)", response, re.DOTALL)
    title = title_match.group(1).strip() if title_match else None

    # Check if the publication year is explicitly provided
    year = None
    year_match = re.search(r"The publication year is (\d{4})", response)
    if year_match:
        year = year_match.group(1)
    else:
        # Check if the year is not provided but the submission date is
        no_year_provided_match = re.search(r"The publication year is not provided\.", response)
        if no_year_provided_match:
            submission_match = re.search(r"The submission happened on .*?(\d{4})", response)
            if submission_match:
                year = submission_match.group(1)
    
    # Fallback for title and year if not found
    if not title:
        title_quotes_match = re.search(r'["“](.*?)["”]', response)
        title_titled_match = re.search(r'titled\s*["“](.*?)["”]', response)
        if title_quotes_match:
            title = title_quotes_match.group(1)
        elif title_titled_match:
            title = title_titled_match.group(1)
        else:
            title = None

    if not year:
        year_fallback_match = re.findall(r'\d{4}', response)
        if year_fallback_match:
            year = max(year_fallback_match)
            print("Warning: Check PDF, year information could be wrong.")

    return authors, title, year

def extract_metadata_new_file(paper_path: str):
    text = read_pdf(paper_path)
    bot, query = initialize_bot(text)
    response = bot.chat(query)
    print(response)
    authors, title, year = preprocess_info(response)
    metadata = {
        'authors': authors,
        'title': title,
        'year' : year
    }
    return metadata
