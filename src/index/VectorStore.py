import os
import time
from logging import Logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.Logger.RAGLogger import RAGLogger
from src.constants import constants


class VectorStore:
    """
    A class to manage the vector store operations, including initialization of Pinecone index
    and creation of the vector store using Huggingface embeddings.

    Attributes:
        log_dir (str): The directory where logs will be stored.
        logger (Logger): Logger instance for logging information and errors.
        pinecone_api_key (str): API key for authenticating with Pinecone.
        pc (Pinecone): Pinecone client instance for interacting with the Pinecone service.
        vector_store (PineconeVectorStore): The vector store instance created using Pinecone.
        embedding_model (HuggingFaceEmbeddings): The embedding model used for embedding the document chunks.
    """

    def __init__(self, logger: Logger = None):
        """
        Initializes the VectorStore instance, setting up the Pinecone client
        and creating the vector store.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        self.logger.info("Initializing VectorStore...")
        self.pinecone_api_key = constants.pinecone_api_key
        self.embedding_model = constants.embedding_model

        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.logger.info("Pinecone client initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing Pinecone client: {e}")
            raise

        self.vector_store = self.create_vectorstore()

    def _initialize_index(self) -> Pinecone.Index:
        """
        Initializes the Pinecone index if it does not already exist.

        Returns:
            Index: The Pinecone index object.
        """
        index_name = "rag-challenge-index"
        self.logger.info(f"Initializing index: {index_name}")

        try:
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

            if index_name not in existing_indexes:
                self.logger.info(f"Creating new index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

                # Wait until the index is ready
                while not self.pc.describe_index(index_name).status["ready"]:
                    self.logger.info("Waiting for the index to be ready...")
                    time.sleep(1)

            index = self.pc.Index(index_name)
            self.logger.info(f"Index {index_name} initialized successfully.")
            return index

        except Exception as e:
            self.logger.error(f"Error initializing index: {e}")
            raise

    def create_vectorstore(self) -> PineconeVectorStore:
        """
        Creates and returns a PineconeVectorStore using Hugging Face embeddings.

        Returns:
            PineconeVectorStore: The vector store instance.
        """
        self.logger.info("Creating vector store with Hugging Face embeddings...")

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            vs_index_name = self._initialize_index()
            vector_store = PineconeVectorStore(index=vs_index_name, embedding=embeddings)
            self.logger.info("Vector store created successfully.")
            return vector_store

        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
