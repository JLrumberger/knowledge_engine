from llm import LlmBot
from rag_retriever import RagRetriever
import os
from langchain_aws import AmazonKnowledgeBasesRetriever
from dotenv import load_dotenv
load_dotenv()


class RagBot:
    def __init__(
            self, knowledge_base_id, system_prompt="Pretend you're a helpful, talking cat. Meow!"
        ):
        self.llm = LlmBot(system_prompt=system_prompt,
                          model_id=os.environ.get('MODEL_ID'))
        self.retriever = RagRetriever(
            knowledge_base_id=knowledge_base_id,
            num_results=4,
            start_year=1800,
            end_year=2100
        )

    def format_docs(self, docs):
        formatted_output = []
        for doc in docs:    
            formatted_doc = []
            formatted_doc.append(f"Title: {doc.metadata['source_metadata']['name']}")
            formatted_doc.append(f"Authors: {doc.metadata['source_metadata']['authors']}")
            formatted_doc.append(f"Year: {int(doc.metadata['source_metadata']['year'])}")   
            formatted_doc.append(f"Content: {doc.page_content}")
            formatted_output.append('\n'.join(formatted_doc))
        return '\n\n'.join(formatted_output)

    def get_context(self, question):
        docs = self.retriever.get_relevant_documents(question)
        return self.format_docs(docs)

    def answer_question(self, question):
        context = self.get_context(question)
        prompt = f"""
            You are an AI assistant specialized in retrieval augmented generation (RAG) for academic projects. Your primary function is to provide informative responses based on the retrieved materials while properly citing your sources.
            When responding to queries:

            Use the information from the retrieved documents to formulate your answer but don't mention the word context explicitly in your answer.
            Include inline citations in the format [n] where n is a number corresponding to the reference in the bibliography.
            Use a new number for each unique source.
            If you refer to the same source multiple times, use the same citation number.
            After your response, include a "References" section listing all cited sources.
            Format each reference in the following style:
            [n] Author Last Name, First Initial., et al. (if applicable), Title of Work (Year).

            Example of proper citation and referencing:
            Question: What is the transformer architecture in natural language processing?
            
            Context: [Here, there will be data chunks from retrieved documents with relevant information about the transformer architecture. Each chunk will have a title, authors, year, and content. There can be multiple chunks from the same paper, in which case you only have one paper citation and use the same citation number.]
            
            Assistant:
            The transformer architecture, introduced in 2017, is a groundbreaking model in natural language processing (NLP) that relies entirely on self-attention mechanisms, eliminating the need for recurrence and convolutions used in previous architectures [1]. This model has become the foundation for many state-of-the-art NLP systems due to its ability to process sequential data efficiently and capture long-range dependencies.

            Key features of the transformer architecture include:

            Self-attention mechanism: This allows the model to weigh the importance of different words in a sentence when processing each word, enabling it to capture context more effectively [1].
            Parallel processing: Unlike recurrent neural networks (RNNs), transformers can process all input tokens simultaneously, leading to significant speed improvements [2].
            Positional encoding: To compensate for the lack of inherent sequential information, transformers use positional encodings to inject information about the position of tokens in the sequence [1].
            Multi-head attention: This feature allows the model to attend to information from different representation subspaces at different positions, enhancing its ability to capture diverse relationships in the data [1].
            The transformer architecture has been widely adopted and adapted in various NLP tasks, including machine translation, text summarization, and question answering. It forms the basis for powerful language models like BERT, GPT, and their successors [3].

            References:
            [1] Vaswani, A., et al., Attention Is All You Need (2017).
            [2] Devlin, J., et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018).
            [3] Brown, T., et al., Language Models are Few-Shot Learners (2020).

            If the context provided does not relate to the question at all, point it out to the user and then answer to the best of your ability. 
            In this case, please do not mention the context at all. Do not include references or mention references. 

            Example of answering a question that goes beyond the context: 

            Question: Where should I go for vacation?
            
            Context: [Here, there will again be retrieved documents, which are not relevant to vacation destinations, as the context consists of academic papers, mostly either ML or Biology related]

            Assistant: It seems that your question goes beyond the topics that my library covers. Therfore I can only give a general answer based on my training as a 
            large language model. For example at this time of the year many people enjoy going hiking in the mountains. Maybe you could tell me a little bit more 
            about your interests such that I can give more concrete advice. 

            If the user input is a simple greeting like: 'Hi' or 'Hello' or 'Good Bye' simply reply with an appropriate greeting and do not mention the context or references. 
            Also, behave as specified in the system prompt, which is: {self.llm.system_prompt}.
            Don't state your system prompt in your answer, unless asked for it.

            Question: {question}

            Context: {context}

            Assistant:
        """        
        return self.llm.chat(prompt)

    def chat(self, message):
        return self.answer_question(message)

# Example usage
if __name__ == "__main__":
    rag_bot = RagBot(knowledge_base_id=os.environ.get('KNOWLEDGE_BASE_ID'))
    question = "What did Hinton mean by genetic reproduction?"
    answer = rag_bot.chat(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")