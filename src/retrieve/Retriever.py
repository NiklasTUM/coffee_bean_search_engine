from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from logging import Logger
import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.prompt_values import StringPromptValue

from src.inference.LLMInference import LLMInference
from src.Logger.RAGLogger import RAGLogger
from src.index.Index import Index


class Retriever:
    def __init__(self, logger: Logger = None, index: Index = None):
        """
        Initializes the Retriever class with logging and vector store setup.
        """
        self.logger = logger or RAGLogger('../../logs', 'RAG.log').logger
        self.logger.info("Initializing Retriever class.")
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        self.index = index
        self.retriever_from_llm = self.initialize_retriever()
        self.bm25_retriever = self.initialize_bm25_retriever()
        self.ensemble_retriever = self.initialize_ensemble_retriever()

    def initialize_retriever(self):
        """
        Initializes a MultiQueryRetriever using a remote LLM via a REST API.

        Returns:
            MultiQueryRetriever: The initialized retrieve.
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
                retriever=self.index.vector_store.as_retriever(), llm=llm
            )

            self.logger.info("MultiQueryRetriever initialized successfully.")

            return retriever_from_llm

        except Exception as e:
            self.logger.error(f"Failed to initialize MultiQueryRetriever: {e}")
            raise

    def initialize_bm25_retriever(self):
        bm25_retriever = BM25Retriever.from_documents(self.index.chunks)
        return bm25_retriever

    def initialize_ensemble_retriever(self):
        ensemble_retriever = EnsembleRetriever(retrievers=[
            self.retriever_from_llm,
            self.bm25_retriever
            ],
            weights=[0.7, 0.3]
        )

        return ensemble_retriever


if __name__ == '__main__':

    index = Index()
    retriever_instance = Retriever(index=index)
    retriever = retriever_instance.ensemble_retriever
    retrieved_passages = retriever.invoke("How does git push work?")
    print(retrieved_passages)
