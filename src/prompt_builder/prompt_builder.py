import os
import yaml
from langchain_core.documents import Document

from src.constants import constants
from src.logger.custom_logger import CustomLogger


class PromptBuilder:
    def __init__(self):
        """
        Initializes the PromptBuilder class, setting up the logger and loading the system prompt.
        """
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, "logs.log").logger
        self.system_prompt_text: str = self._load_system_prompt_text()

        self.logger.info("PromptBuilder initialized successfully with system prompt.")

    def _load_system_prompt_text(self) -> str:
        """
        Loads the system prompt text from the specified YAML file.

        Returns:
            str: The system prompt text loaded from the YAML file.
        """
        try:
            self.logger.info(f"Loading system prompt text from {constants.prompt_template_path}...")
            with open(constants.prompt_template_path, 'r', encoding='utf-8') as file:
                prompt_data = yaml.safe_load(file)
                system_prompt = prompt_data.get("system_role")  # Use .get for safer access
                if not system_prompt or not isinstance(system_prompt, str):
                    raise ValueError("System prompt ('system_role') not found or is not a string in YAML.")
            self.logger.info("System prompt text loaded successfully.")
            return system_prompt
        except Exception as e:
            self.logger.error(f"Error loading system prompt text: {e}")
            raise

    def get_system_prompt(self) -> str:
        """
        Returns the loaded system prompt text.
        """
        return self.system_prompt_text

    def _convert_document_to_coffee_json_string(self, doc: Document) -> str:
        """
        Converts a LangChain Document to a structured JSON-like string for prompting.

        Args:
            doc (Document): A LangChain Document containing coffee data.

        Returns:
            str: A string representation of the cleaned coffee data.
        """
        filtered_metadata = {
            k: v for k, v in doc.metadata.items()
            if k not in {"desc_2", "desc_3"}  # Example filter
        }

        # Construct a readable string. Using YAML dump for nice formatting,
        # but you can format it any way you prefer.
        coffee_data_dict = {
            "flavor_description": doc.page_content,
            **filtered_metadata
        }
        return yaml.dump(coffee_data_dict, sort_keys=False, allow_unicode=True)

    def create_user_content(self, query: str, search_result: Document) -> list[str]:
        """
        Creates the user content parts for the LLM, combining the query and context.
        This content will be passed as the user's message to the Gemini API.

        Args:
            query (str): The user's query.
            search_result (Document): A single search result (e.g., Langchain Document).

        Returns:
            list[str]: A list of strings representing parts of the user's message.
                       This allows for multi-part messages if needed (e.g., text and image),
                       or can be joined into a single string if preferred for text-only.
        """
        coffee_description_str = self._convert_document_to_coffee_json_string(search_result)

        user_content_parts = [
            f"User Query: {query}",
            f"Relevant Coffee Information:\n---\n{coffee_description_str}\n---"
        ]

        self.logger.info(f"Created user content parts with query and document context.")
        return user_content_parts
