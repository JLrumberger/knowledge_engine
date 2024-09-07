from llm import LlmBot
from langchain_aws import AmazonKnowledgeBasesRetriever

class RagBot:
    def __init__(self, knowledge_base_id, system_prompt="Pretend you're a helpful, talking cat. Meow!"):
        self.bot = LlmBot(system_prompt=system_prompt)
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 4,
                }
            }
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(self, question):
        docs = self.retriever.get_relevant_documents(question)
        return self.format_docs(docs)

    def answer_question(self, question):
        context = self.get_context(question)
        prompt = f"""Answer the question based on the context provided. Also, adhere to the system prompt.

        Context: {context}

        Question: {question}"""
        
        return self.bot.chat(prompt)

    def chat(self, message):
        return self.answer_question(message)

# Example usage
if __name__ == "__main__":
    import os
    os.environ['AWS_ACCESS_KEY_ID'] = ''
    os.environ['AWS_SECRET_ACCESS_KEY'] = ''
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'
    os.environ['AWS_SESSION_TOKEN'] = ''    
    rag_bot = RagBot(knowledge_base_id="4FYUGYITNF")
    question = "What did Hinton mean by genetic reproduction?"
    answer = rag_bot.chat(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")