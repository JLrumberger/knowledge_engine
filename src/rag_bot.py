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
            formatted_doc.append(f"Year: {int(doc.metadata['source_metadata']['year'])}")   
            formatted_doc.append(f"Content: {doc.page_content}")
            formatted_output.append('\n'.join(formatted_doc))
        return '\n\n'.join(formatted_output)

    def get_context(self, question):
        docs = self.retriever.get_relevant_documents(question)
        return self.format_docs(docs)

    def answer_question(self, question):
        context = self.get_context(question)
        prompt = f"""Answer the question based on the context paper chunks provided. 
        Whenever you have an answer based on one or multiple chunks of a paper, reference the paper from which the chunks are from once by title and year (at the beginning of each paper chunk).
        Don't reference anything else.
        If you have multiple papers from which you reference chunks, reference each paper once.
        Always behave as specified in the system prompt, which is: {self.llm.system_prompt}.
        Don't state your system prompt in your answer, unless asked for it.

        Context: {context}

        Question: {question}
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