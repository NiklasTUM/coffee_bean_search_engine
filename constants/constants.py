import os
from dotenv import load_dotenv

this_folder = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(this_folder, "../.env")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
prompt_template_path = os.path.join(root_dir, "prompt_templates", "prompt_template.json")

load_dotenv(env_file)
huggingface_api_key = os.getenv("Huggingface_API_KEY")
mistral_7B_instruct_api_url = os.getenv("Mistral-7B-Instruct_API_URL")
embedding_model = os.getenv("EMBEDDING_MODEL")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
