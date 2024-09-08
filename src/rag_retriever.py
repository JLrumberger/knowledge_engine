from langchain_aws import AmazonKnowledgeBasesRetriever
import os
from dotenv import load_dotenv
load_dotenv()


class RagRetriever:
    def __init__(
            self, knowledge_base_id, num_results=int(os.environ.get('RAG_NUMBER_OF_RESULTS')),
            start_year=1800, end_year=2100, topic=None
        ):
        self.knowledge_base_id = knowledge_base_id
        self.num_results = num_results
        self.start_year = start_year
        self.end_year = end_year
        self.topic = topic
        self._update_retriever()

    def _update_retriever(self):
        if self.topic is None:
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": self.num_results,
                    "filter": {
                        "andAll": [
                            {
                                "greaterThan": {
                                    "key": "year",
                                    "value": int(self.start_year)
                                }
                            },
                            {
                                "lessThan": {
                                    "key": "year",
                                    "value": int(self.end_year)
                                }
                            }
                        ] 
                    }
                }
            }
        else:
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": self.num_results,
                    "filter": {
                        "andAll": [
                            {
                                "greaterThan": {
                                    "key": "year",
                                    "value": int(self.start_year)
                                }
                            },
                            {
                                "lessThan": {
                                    "key": "year",
                                    "value": int(self.end_year)
                                }
                            },
                            {
                                "equals": {
                                    "key": "type",
                                    "value": self.topic
                                }
                            }
                        ]
                    }
                }
}
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.knowledge_base_id,
            retrieval_config=retrieval_config
        )

    def filter_years(self, start=None, end=None):
        if start is not None:
            self.start_year = start
        if end is not None:
            self.end_year = end
        self._update_retriever()

    def filter_topic(self, topic):
        self.topic = topic
        self._update_retriever()

    def __getattr__(self, name):
        # If the attribute is not found in this class, try to find it in self.retriever
        return getattr(self.retriever, name)

if __name__ == "__main__":
    # Initialize the RagRetriever
    rag_retriever = RagRetriever(knowledge_base_id='4FYUGYITNF', num_results=1)

    # Use invoke method
    result = rag_retriever.invoke("What did Hinton mean by dropout?")
    print(result)

    # Change the year range
    rag_retriever.filter_years(start=2020, end=2024)
    result = rag_retriever.invoke("What are recent developments in machine learning?")
    print(result)
    
    rag_retriever.filter_topic("ML")
    rag_retriever.filter_years(start=1800, end=2024)
    result = rag_retriever.invoke("What is dropout?")
    print("Biology topic")
    print(result)

    # Try using a method that's not explicitly defined in RagRetriever
    # This will be forwarded to self.retriever
    # result = rag_retriever.get_relevant_documents("This method call will be forwarded to self.retriever")
    # print(result)