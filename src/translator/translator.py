import os

from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from langchain_core.documents import Document

from src.constants import constants


class Translator:

    def translate_text(self, text: str, target_language: str = "en") -> dict:
        credentials = service_account.Credentials.from_service_account_file(
            filename=os.path.join(constants.root_dir, "src", "translator", "google_service_credentials.json")
        )
        client = translate.Client(credentials=credentials)
        result = client.translate(text, target_language=target_language)

        return {
            "translated_text": result['translatedText'],
            "detected_source_language": result.get('detectedSourceLanguage', 'unknown')
        }

    def translate_document_fields(self, doc: Document, target_language: str = "en") -> Document:
        # Translate main text
        translated_main = self.translate_text(doc.page_content, target_language=target_language)

        # Translate metadata values if they are strings
        translated_metadata = {}
        for key, value in doc.metadata.items():
            if key == "desc_2" or key == "desc_3":
                continue

            if isinstance(value, str):
                result = self.translate_text(value, target_language=target_language)
                translated_metadata[key] = result["translated_text"]
            else:
                translated_metadata[key] = value  # Keep non-string values untouched

        # Return new translated Document
        return Document(
            page_content=translated_main["translated_text"],
            metadata=translated_metadata
        )


if __name__ == "__main__":
    # Example usage
    translator = Translator()
    text = "Crisply sweet-tart. Green apple, magnolia, oak, maple syrup, almond\n  in aroma and cup."
    translated = translator.translate_text(text, target_language="de")
    print(translated)
