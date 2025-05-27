import logging
import os
from logging import Logger
from typing import List, Dict
from huggingface_hub import InferenceClient
import time
import requests
from src.logger.custom_logger import CustomLogger
from src.constants import constants
import httpx  
from requests.exceptions import RequestException
from ssl import SSLError

class LLMInference:
    """
    A class to generate answers using a language model via the Hugging Face Inference API.

    Attributes:
        log_dir (str): The directory where logs will be stored.
        logger (Logger): logger instance for logging information and errors.
        api_key (str): API key for the Hugging Face Inference API.
        api_url (str): The URL of the language model hosted on Hugging Face.
        client (InferenceClient): The InferenceClient instance to interact with the Hugging Face API.
    """

    def __init__(self, logger: Logger = None):
        """
        Initializes the AnswerGenerator instance with a logger, API key, and URL.

        Args:
            logger (logger, optional): logger instance for logging. If not provided, a default logger is set up.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or CustomLogger(self.log_dir, "RAG.log").logger
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

    # def inference(self, prompt: List[Dict[str, str]]) -> str:
    #     """
    #     Generates a response from the language model based on the provided prompt.

    #     Args:
    #         prompt (list[dict]): A list of dictionaries representing the single prompt.

    #     Returns:
    #         str: The generated response from the language model.
    #     """
    #     max_retries = 3
    #     retry_delay = 2  # seconds

    #     for attempt in range(max_retries):
    #         try:
    #             self.logger.info(f"Starting inference (attempt {attempt + 1})")
    #             output = ""
    #             for message in self.client.chat_completion(
    #                     messages=prompt,
    #                     max_tokens=8192,
    #                     stream=True,
    #             ):
    #                 output += message.choices[0].delta.content
    #             self.logger.info("Inference completed successfully.")
    #             return output
    #         except (requests.exceptions.RequestException, Exception) as e:
    #             self.logger.error(f"Inference failed (attempt {attempt + 1}): {e}")
    #             if attempt < max_retries - 1:
    #                 time.sleep(retry_delay)
    #                 retry_delay *= 2  # exponential backoff
    #             else:
    #                 self.logger.error("Max retries exceeded. Giving up.")
    #                 raise
    def inference(self, prompt: List[Dict[str, str]]) -> str:
        """
        Generates a response from the language model with robust retry logic on network failures.
        """
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Starting inference (attempt {attempt})")

                output = ""
                # Actual API call (streamed)
                for message in self.client.chat_completion(
                    messages=prompt,
                    max_tokens=8192,
                    stream=True,
                ):
                    output += message.choices[0].delta.content

                self.logger.info("Inference completed successfully.")
                return output

            except (httpx.HTTPError, RequestException, ConnectionError, SSLError) as e:
                self.logger.error(f"Inference failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    self.logger.error("Max retries exceeded. Giving up.")
                    raise

            except Exception as e:
                # Catches unexpected things like JSON decode or attribute errors
                self.logger.error(f"Unexpected error during inference (attempt {attempt}): {e}")
                raise


if __name__ == "__main__":
    try:
        answer_generator = LLMInference()
        example_prompt = [{"role": "user", "content": "What is the capital of France?"}]
        answer = answer_generator.inference(example_prompt)
        print(answer)
    except Exception as exc:
        logging.error(f"Failed to generate answer: {exc}")
