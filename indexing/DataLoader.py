import os
import logging

from langchain_core.documents import Document
from constants import constants
from RAGLogger import RAGLogger
from langchain_community.document_loaders import UnstructuredMarkdownLoader

import nltk


class DataLoader:
    def __init__(self, directory: str, logger: logging.Logger = None):
        """
        Initializes the DataLoader with the specified directory and a logger.

        Args:
            directory (str): The directory from which to load markdown files.
            logger (Logger, optional): Logger instance for logging. If not provided, a default logger is set up.
        """
        self.directory = directory
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')

    def load_data(self) -> list[Document]:
        """
        Loads markdown files from the directory and returns their content.

        Returns:
            list[Document]: A list of documents, where each document contains the content of a markdown file.
        """
        self.logger.info(f"Loading data from directory: {self.directory}")
        try:
            markdown_files = self._get_markdown_files()
            if not markdown_files:
                self.logger.warning(f"No markdown files found in directory: {self.directory}")
            data = [self._read_file_custom(file_path) for file_path in markdown_files]
            self.logger.info(f"Successfully loaded {len(data)} markdown files.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _get_markdown_files(self) -> list:
        try:
            files = [os.path.join(self.directory, file) for file in os.listdir(self.directory)
                     if file.endswith('.md')]
            self.logger.info(f"Found {len(files)} markdown files.")
            return files
        except Exception as e:
            self.logger.error(f"Error while retrieving markdown files: {e}")
            raise

    def _read_file_custom(self, file_path: str) -> Document:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.info(f"Successfully read file: {file_path}")
            return Document(page_content=content, metadata={'source': file_path})
        except Exception as e:
            self.logger.error(f"Error while reading file {file_path}: {e}")
            raise

    def _read_file_langchain(self, file_path: str) -> Document:
        try:
            md_doc = UnstructuredMarkdownLoader(file_path)
            loaded_data = md_doc.load()
            self.logger.info(f"Successfully read file: {file_path}")
            return loaded_data[0]
        except Exception as e:
            self.logger.error(f"Error while reading file {file_path}: {e}")
            raise


if __name__ == "__main__":
    logger_instance = RAGLogger().logger
    data_path = os.path.join(constants.root_dir, "data")
    data_loader = DataLoader(data_path, logger=logger_instance)
    markdown_data = data_loader.load_data()

    for index, content in enumerate(markdown_data):
        print(f"Content of file {index + 1}:\n{content}...\n")
