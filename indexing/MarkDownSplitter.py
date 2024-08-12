import logging
import os

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from RAGLogger import RAGLogger
from indexing.DataLoader import DataLoader

from constants import constants


class MarkDownSplitter:
    """
    A class to handle the splitting of Markdown documents into smaller chunks
    based on headers and character limits.

    Attributes:
        log_dir (str): The directory where logs will be stored.
        logger (Logger): Logger instance for logging information and errors.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the MarkDownSplitter with an optional logger.

        Args:
            logger (Logger, optional): Logger instance for logging. If not provided, a default logger is set up.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        nltk.download('punkt')
        self.logger.info("MarkDownSplitter initialized successfully.")

    def header_splitter(self, markdown_document: Document) -> list[Document]:
        """
        Splits a Markdown document based on specified headers and retains metadata.

        Args:
            markdown_document (Document): The original Markdown document to be split.

        Returns:
            list[Document]: A list of Document objects, each representing a split part of the original document,
                            with metadata retained.
        """
        self.logger.info("Starting header-based splitting of the markdown document.")
        try:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            md_header_splits = markdown_splitter.split_text(markdown_document.page_content)

            # Retain metadata and create new Document objects for each split
            split_documents = [
                Document(page_content=content.page_content, metadata=markdown_document.metadata)
                for content in md_header_splits
            ]

            self.logger.info(f"Header-based splitting produced {len(split_documents)} chunks.")
            return split_documents
        except Exception as e:
            self.logger.error(f"Error during header-based splitting: {e}")
            raise

    def character_splitter(self, chunk: Document) -> list[Document]:
        """
        Further splits a document chunk into smaller pieces based on character limits.

        Args:
            chunk (Document): A Document chunk to be split further.

        Returns:
            list[Document]: A list of Document objects split based on character limits.
        """
        self.logger.info("Starting character-based splitting of a document chunk.")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            splitted_docs = text_splitter.create_documents([chunk.page_content], metadatas=[chunk.metadata])
            self.logger.info(f"Character-based splitting produced {len(splitted_docs)} chunks.")
            return splitted_docs
        except Exception as e:
            self.logger.error(f"Error during character-based splitting: {e}")
            raise

    def hybrid_split(self, markdown_document: Document) -> list[Document]:
        """
        Splits a Markdown document first by headers and then recursively by character limits.

        Args:
            markdown_document (Document): The Markdown document to be split.

        Returns:
            list[Document]: A list of final document chunks after hierarchical and recursive character splitting.
        """
        self.logger.info("Starting hybrid splitting (header-based and character-based).")
        try:
            hierarchical_chunks = self.header_splitter(markdown_document)
            final_chunks = []
            for chunk in hierarchical_chunks:
                final_chunks.extend(self.character_splitter(chunk))
            self.logger.info(f"Hybrid splitting produced {len(final_chunks)} chunks.")
            return final_chunks
        except Exception as e:
            self.logger.error(f"Error during hybrid splitting: {e}")
            raise


if __name__ == "__main__":
    logger_instance = RAGLogger().logger

    # Instantiate the splitter
    splitter = MarkDownSplitter(logger=logger_instance)
    data_loader = DataLoader('../data', logger_instance)
    data = data_loader.load_data()
    md_doc = data[0]

    try:
        chunks = splitter.hybrid_split(md_doc)
        for i, doc_chunk in enumerate(chunks):
            print(f"Chunk {i + 1}: {doc_chunk.page_content[:100]}...")
    except Exception as exc:
        logger_instance.error(f"Error during splitting process: {exc}")
