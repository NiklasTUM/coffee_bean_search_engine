from logging import Logger
import os
from pathlib import Path

from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from src.logger.RAGLogger import RAGLogger
from src.constants import constants
from src.index.DataLoader import DataLoader
from src.index.MarkDownSplitter import MarkDownSplitter
from src.index.VectorStore import VectorStore


class Index:
    """
    A class to manage the index of documents, including loading data,
    splitting markdown files into chunks, and adding those chunks to an index.

    Attributes:
        log_dir (str): Directory where logs are stored.
        logger (RAGLogger): logger instance for logging information and errors.
        record_manager (SQLRecordManager): Manages the records in the SQL database.
        vector_store (PineconeVectorStore): Vector store instance for managing embeddings.
        data_path (str): Path to the data directory.
        data_loader (DataLoader): Instance to load data from the data directory.
        markdown_splitter (MarkDownSplitter): Instance to split Markdown documents into chunks.
        chunks (list[Document]): The list of document chunks after splitting/chunking the source documents.
    """

    def __init__(self, logger: Logger = None):
        """
        Initializes the Index class, setting up the logger, record manager, vector store,
        data loader, and markdown splitter.

        Args:

            logger (logger, optional): logger instance for logging. If not provided, a new RAGLogger is set up.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        self.logger.info("Initializing Index class...")
        self.data_path = os.path.join(constants.root_dir, "data")
        self.data_loader = DataLoader(self.data_path, self.logger)
        self.markdown_splitter = MarkDownSplitter(self.logger)

        self.record_manager = self.initialize_record_manager(self.logger)
        self.vector_store = VectorStore().create_vectorstore()
        self.chunks = self.store_chunks()

        self.logger.info("Index class initialized successfully.")

    @staticmethod
    def initialize_record_manager(logger: Logger) -> SQLRecordManager:
        """
        Initializes the SQLRecordManager with a specified namespace and database URL.

        Returns:
            SQLRecordManager: The initialized SQLRecordManager instance.
        """
        namespace = "RAGChallenge"
        db_dir = os.path.join(constants.root_dir, "db", "record_manager_cache.sql")
        db_path = Path(db_dir).as_posix()
        db_url = f'sqlite:///{db_path}'
        record_manager = SQLRecordManager(
            namespace, db_url=db_url
        )

        record_manager.create_schema()
        logger.info(f"SQLRecordManager initialized with namespace: {namespace}")
        return record_manager

    def add_chunk_to_index(self, chunk_doc: list[Document]):
        """
        Adds a list of document chunks to the index.

        Args:
            chunk_doc (list[Document]): A list of Document objects to be added to the index.
        """
        try:
            self.logger.info("Adding document chunks to the index...")
            index(
                chunk_doc,
                self.record_manager,
                self.vector_store,
                cleanup="incremental",
                source_id_key="source"
            )
            self.logger.info("Document chunks added to the index successfully.")
        except Exception as e:
            self.logger.error(f"Failed to add chunks to the index: {e}")
            raise

    def index_documents(self):
        """
        Loads Markdown documents, splits them into chunks, and indexes those chunks.
        """
        try:
            self.logger.info("Starting the document index process...")
            loaded_documents = self.data_loader.load_data()
            chunks = []
            for doc in loaded_documents:
                chunks.extend(self.markdown_splitter.hybrid_split(doc))

            self.chunks = chunks
            self.add_chunk_to_index(chunks)
            self.logger.info("Document index process completed successfully.")
        except Exception as e:
            self.logger.error("An error occurred during the index process.")
            self.logger.error(f"Error: {e}")
            raise

    def store_chunks(self):
        loaded_documents = self.data_loader.load_data()
        chunks = []
        for doc in loaded_documents:
            chunks.extend(self.markdown_splitter.hybrid_split(doc))

        return chunks


if __name__ == '__main__':
    rag_logger = RAGLogger().logger
    data_path = os.path.join(constants.root_dir, "data")
    data_loader = DataLoader(data_path, logger=rag_logger)
    markdown_splitter = MarkDownSplitter(rag_logger)
    my_index = Index(rag_logger)

    # Perform the index
    try:
        my_index.index_documents()
    except Exception as exc:
        rag_logger.error(f"Error during the document index process: {exc}")
