import os
from dotenv import load_dotenv

this_folder = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(this_folder, "../.env")

load_dotenv(env_file)
huggingface_api_key = os.getenv("Huggingface_API_KEY")
mistral_7B_instruct_api_url = os.getenv("Mistral-7B-Instruct_API_URL")
bge_embedding_model_api_url = os.getenv("BGE-M3_EMBEDDING_MODEL_API_URL")
pinecone_api_key=os.getenv("PINECONE_API_KEY")
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
prompt_template_path = os.path.join(root_dir, "prompt_templates", "prompt_template.json")