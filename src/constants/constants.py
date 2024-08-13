import os
from dotenv import load_dotenv


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
env_file = os.path.join(root_dir, ".env")
prompt_template_path = os.path.join(root_dir, "src", "prompt_templates", "prompt_template.json")

load_dotenv(env_file)
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
mistral_7B_instruct_api_url = os.getenv("MISTRAL_7B_INSTRUCT_API_URL")
embedding_model = os.getenv("EMBEDDING_MODEL")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
