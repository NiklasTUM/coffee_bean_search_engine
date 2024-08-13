import json
import os

from src.inference.LLMInference import LLMInference
from src.logger.RAGLogger import RAGLogger
from src.retrieve.Retriever import Retriever
from src.constants import constants
from src.index.Index import Index


class RAGChain:
    """
    A class to manage the entire RAG (Retrieval-Augmented Generation) process,
    including retrieving relevant documents, generating answers, and updating the index.

    Attributes:
        logger (logger): logger instance for logging information and errors.
        retriever (Retriever): Instance of the Retriever class for document retrieval.
        answer_generator (LLMInference): Instance of LLMInference for generating answers.
        system_prompt (str): The system prompt loaded from the JSON file.
        index (Index): Instance of the Index class for managing the document index.
    """

    def __init__(self):
        """
        Initializes the RAGChain class, setting up the logger, retrieve,
        answer generator, system prompt, and index.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = RAGLogger(self.log_dir, "RAG.log").logger
        self.logger.info("Initializing RAGChain...")

        self.index = Index(self.logger)
        self.retriever = Retriever(self.logger, self.index)
        self.answer_generator = LLMInference(self.logger)
        self.system_prompt = self.load_system_prompt()

        self.logger.info("RAGChain initialized successfully.")

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt from the specified JSON file.

        Returns:
            str: The system prompt loaded from the JSON file.

        Raises:
            ValueError: If the system prompt is not found in the JSON file.
            Exception: If there is an error loading the system prompt.
        """
        try:
            self.logger.info(f"Loading system prompt from {constants.prompt_template_path}...")
            with open(constants.prompt_template_path, 'r', encoding='utf-8') as file:
                prompt_data = json.load(file)
                self.system_prompt = prompt_data.get("system")
                if not self.system_prompt:
                    raise ValueError("System prompt not found in the JSON file.")
            self.logger.info("System prompt loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading system prompt: {e}")
            raise

        return self.system_prompt

    @staticmethod
    def format_docs(docs) -> str:
        """
        Formats a list of documents into a single string by concatenating their contents.

        Args:
            docs (list[Document]): A list of Document objects.

        Returns:
            str: A single string containing the concatenated contents of the documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def create_prompt(self, context: str, question: str) -> list[dict]:
        """
        Creates a prompt for the LLM by combining the system prompt, context, and question.

        Args:
            context (str): The retrieved context to include in the prompt.
            question (str): The user's question to include in the prompt.

        Returns:
            list[dict]: A list of dictionaries representing the LLM prompt.
        """
        prompt = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
        self.logger.info(f"Prompt created for the question: {question}")
        return prompt

    def chain(self, question: str) -> str:
        """
        Orchestrates the full RAG process: retrieves context, creates a prompt,
        generates an answer, and logs the answer.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        try:
            self.logger.info(f"Processing question: {question}")
            retrieved_context = self.retriever.ensemble_retriever.invoke(question)
            formatted_context = self.format_docs(retrieved_context)
            self.logger.info(f"Retrieved {len(retrieved_context)} chunks.")
            self.logger.info(f"Retrieved Context: {formatted_context}")
            prompt = self.create_prompt(formatted_context, question)
            answer = self.answer_generator.inference(prompt)
            self.logger.info(f"Generated answer: {answer}")
            return answer
        except Exception as e:
            self.logger.error(f"Error during the chain process: {e}")
            raise

    def update_index(self):
        """
        Updates the document index by reloading the data, splitting it into chunks, and index those chunks.
        """
        try:
            self.logger.info("Updating the document index...")
            self.index.index_documents()
            self.logger.info("Document index updated successfully.")
        except Exception as e:
            self.logger.error(f"Error updating the document index: {e}")
            raise


if __name__ == '__main__':
    rag_chain = RAGChain()
    example_question = "How does git push work?"
    try:
        rag_chain.chain(example_question)
    except Exception as exc:
        rag_chain.logger.error(f"Error in main execution: {exc}")
