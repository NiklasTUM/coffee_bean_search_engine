import os
from dotenv import load_dotenv


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
env_file = os.path.join(root_dir, ".env")
prompt_template_path = os.path.join(root_dir, "src", "prompt_templates", "system_prompt.yaml")

load_dotenv(env_file)
gemini_api_key = os.getenv("GEMINI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_model_ml = os.getenv("EMBEDDING_MODEL_ML")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_translate_api_key = os.getenv("GOOGLE_API_KEY")
es_url = os.getenv("ES_URL")
es_api_key = os.getenv("ES_API_KEY")
es_index_name = os.getenv("ES_INDEX_NAME")