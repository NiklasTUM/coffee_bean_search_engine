import os

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.constants import constants
from src.logger.custom_logger import CustomLogger


class LLMInference:
    """
    A class to generate content using a language model via the Google Gemini API,
    leveraging system instructions and structured user prompts.
    """

    def __init__(self, system_instruction: str, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the LLMInference instance.

        Args:
            system_instruction (str): The system-level instruction for the model.
            model_name (str, optional): The name of the Gemini model to use.
                                         Defaults to "gemini-1.5-flash-latest".
                                         Other options: "gemini-1.0-pro", "gemini-1.5-pro-latest".
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, "logs.log").logger

        try:
            genai.configure(api_key=constants.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    top_p=1.0,
                    max_output_tokens=2048,
                ),
            )
            self.logger.info(f"GenerativeModel '{model_name}' initialized successfully with system instruction.")
        except Exception as e:
            self.logger.error(f"Error initializing GenerativeModel: {e}")
            raise

    def inference(self, user_content: list[str] | str) -> str:
        """
        Generates a response from the language model based on the provided user content.
        The system instruction is applied at the model level.

        Args:
            user_content (list[str] | str):
                A list of strings for a multi-part user message (e.g., query + context parts),
                or a single string for a simple user message.

        Returns:
            str: The generated text response from the language model.
                 Returns an error message string if generation is blocked or fails.
        """
        try:
            self.logger.info(f"Starting inference with user content: {str(user_content)[:200]}...")  # Log snippet

            response = self.model.generate_content(
                contents=user_content,
            )

            # Basic check for blocked response (more robust error handling might be needed)
            if not response.candidates or not response.candidates[0].content.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_message = response.prompt_feedback.block_reason_message or "Unknown reason"
                    self.logger.warning(
                        f"Inference blocked. Reason: {response.prompt_feedback.block_reason}, Message: {block_reason_message}")
                    return f"Response blocked due to: {block_reason_message}"
                self.logger.warning("Inference returned no content or was possibly blocked without detailed feedback.")
                return "Response could not be generated (possibly blocked or empty)."

            self.logger.info("Inference completed successfully.")
            return response.text

        except Exception as e:  # Catching broader exceptions from the API call
            self.logger.error(f"Error during Gemini API inference: {e}", exc_info=True)
            # You might want to check for specific google.api_core.exceptions
            return f"An error occurred during inference: {str(e)}"

