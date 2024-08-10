import os
from dotenv import load_dotenv

this_folder = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(this_folder, "../.env")

load_dotenv(env_file)
huggingface_api_key = os.getenv("Huggingface_API_KEY")
mistral_7B_instruct_api_url = os.getenv("Mistral-7B-Instruct_API_URL")
