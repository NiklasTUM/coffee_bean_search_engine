import logging

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from RAGLogger import RAGLogger
from indexing.DataLoader import DataLoader


class MarkDownSplitter:
    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the MarkDownSplitter with an optional logger.

        Args:
            logger (Logger, optional): Logger instance for logging. If not provided, a default logger is set up.
        """
        self.logger = logger or RAGLogger().logger
        nltk.download('punkt')

    def header_splitter(self, markdown_document: Document) -> list[Document]:
        """
        Splits a markdown document based on specified headers and retains metadata.

        Args:
            markdown_document (Document): The original markdown document to be split.

        Returns:
            list[Document]: A list of Document objects, each representing a split part of the original document,
                            with metadata retained.
        """
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

        return split_documents

    def character_splitter(self, chunk: Document) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        data = text_splitter.create_documents([chunk.page_content], metadatas=[chunk.metadata])
        return data

    def hybrid_split(self, markdown_document: Document) -> list[Document]:
        """
        Splits markdown text first hierarchically, then semantically.

        Args:
            markdown_text (str): The markdown text to be split.

        Returns:
            list: A list of final chunks after hierarchical and semantic splitting.
        """
        self.logger.info("Starting full markdown splitting (hierarchical and semantic).")
        try:
            hierarchical_chunks = self.header_splitter(markdown_document)
            final_chunks = []
            for chunk in hierarchical_chunks:
                final_chunks.extend(self.character_splitter(chunk))
            self.logger.info(f"Final split produced {len(final_chunks)} chunks.")
            return final_chunks
        except Exception as e:
            self.logger.error(f"Error during markdown splitting: {e}")
            raise


if __name__ == "__main__":
    logger_instance = RAGLogger().logger

    # Instantiate the splitter
    splitter = MarkDownSplitter(logger=logger_instance)
    data_loader = DataLoader('../data', logger_instance)
    data = data_loader.load_data()
    md_doc = data[0]

    try:
        final_chunks = splitter.hybrid_split(md_doc)
        for i, chunk in enumerate(final_chunks):
            print(f"Chunk {i + 1}: {chunk[:100]}...")  # Print the first 100 characters of each chunk as a preview
    except Exception as e:
        logger_instance.error(f"Error during splitting process: {e}")
