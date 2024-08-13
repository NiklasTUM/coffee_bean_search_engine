import logging
import os
from logging import Logger
from typing import List, Dict
from huggingface_hub import InferenceClient

from src.Logger.RAGLogger import RAGLogger
from src.constants import constants


class LLMInference:
    """
    A class to generate answers using a language model via the Hugging Face Inference API.

    Attributes:
        log_dir (str): The directory where logs will be stored.
        logger (Logger): Logger instance for logging information and errors.
        api_key (str): API key for the Hugging Face Inference API.
        api_url (str): The URL of the language model hosted on Hugging Face.
        client (InferenceClient): The InferenceClient instance to interact with the Hugging Face API.
    """

    def __init__(self, logger: Logger = None):
        """
        Initializes the AnswerGenerator instance with a logger, API key, and URL.

        Args:
            logger (Logger, optional): Logger instance for logging. If not provided, a default logger is set up.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or RAGLogger(self.log_dir, "RAG.log").logger
        self.api_key = constants.huggingface_api_key
        self.api_url = constants.mistral_7B_instruct_api_url
        self.client = self._initialize_client()

    def _initialize_client(self) -> InferenceClient:
        """
        Initializes the Hugging Face InferenceClient with the provided API key and URL.

        Returns:
            InferenceClient: The initialized InferenceClient object.
        """
        try:
            client = InferenceClient(
                model=self.api_url,
                token=self.api_key,
            )
            self.logger.info("InferenceClient initialized successfully.")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize InferenceClient: {e}")
            raise

    def inference(self, prompt: List[Dict[str, str]]) -> str:
        """
        Generates a response from the language model based on the provided prompt.

        Args:
            prompt (list[dict]): A list of dictionaries representing the single prompt.

        Returns:
            str: The generated response from the language model.
        """
        try:
            self.logger.info(f"Starting inference")
            output = ""
            for message in self.client.chat_completion(
                    messages=prompt,
                    max_tokens=8192,
                    stream=True,
            ):
                output += message.choices[0].delta.content
            self.logger.info("Inference completed successfully.")
            return output
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise


if __name__ == "__main__":
    try:
        answer_generator = LLMInference()
        example_prompt = [{"role": "user", "content": "What is the capital of France?"}]
        answer = answer_generator.inference(example_prompt)
        print(answer)
    except Exception as exc:
        logging.error(f"Failed to generate answer: {exc}")
