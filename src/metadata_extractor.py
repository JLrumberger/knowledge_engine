import logging
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

def initialize_bot(text: str):
    prompt = "Pretend to be a helpful research assistant"
    query = f"What are the authors' names and title contained in this first page of paper: {text}? \n" \
        "And do you know the publication year? \n" \
        "In addition, you can always respond in the same format:\n" \
        "Authors: [...] \n" \
        "Title: [...] \n" \
        "The publication year [...]. The submission happened on [...] \n" \
        "If there is neither publication year nor submission, reply with  \n" \
        "The publication year is not provided [...] \n" \
        "Then \n The submission year happened on [...]" \
        "If none of the above applies, please state a guess year only if there is a clear and possible reference in the text" \
        "(e.g., a footnote indicating the 31st Conference on Neural Information Processing Systems (NIPS 2017) or" \
        "a time stamp from arxiv). Otherwise, if you specify that it appears to be only a" \
        "recent publication, avoid providing the current or previous year."
        
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
