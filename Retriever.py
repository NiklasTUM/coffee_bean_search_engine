from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.prompt_values import StringPromptValue

from LLMInference import LLMInference
from RAGLogger import RAGLogger
from indexing.VectorStore import VectorStore


class Retriever:
    def __init__(self, logger: RAGLogger = None):
        """
        Initializes the Retriever class with logging and vector store setup.
        """
        self.logger = logger or RAGLogger('logs', 'RAG.log').logger
        self.logger.info("Initializing Retriever class.")
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        self.vector_store = VectorStore().create_vectorstore()
        self.retriever_from_llm = self.initialize_retriever()

    def initialize_retriever(self):
        """
        Initializes a MultiQueryRetriever using a remote LLM via a REST API.

        Returns:
            MultiQueryRetriever: The initialized retriever.
        """
        try:
            self.logger.info("Initializing MultiQueryRetriever.")
            logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

            def llm(prompt) -> str:
                self.logger.info(f"Processing prompt: {prompt}")
                if isinstance(prompt, StringPromptValue):
                    prompt_text = prompt.text
                else:
                    prompt_text = str(prompt)

                formatted_prompt = [{"role": "user", "content": prompt_text}]
                response = LLMInference().inference(formatted_prompt)
                return response

            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=self.vector_store.as_retriever(), llm=llm
            )

            self.logger.info("MultiQueryRetriever initialized successfully.")

            return retriever_from_llm

        except Exception as e:
            self.logger.error(f"Failed to initialize MultiQueryRetriever: {e}")
            raise


if __name__ == '__main__':
    try:
        retriever_instance = Retriever()
        retriever = retriever_instance.retriever_from_llm
        generated_queries = retriever.invoke("What are the approaches to Task Decomposition?")
        print(generated_queries)
    except Exception as e:
        logging.error(f"Error during retrieval process: {e}")
