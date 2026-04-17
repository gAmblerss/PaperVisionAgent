
from langchain.chat_models import init_chat_model
from openai import OpenAI
import os

def build_chat_model(model_name,url):
    model = init_chat_model(
        # model_name = model_name,
        model_name,
        model_provider="openai",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url = url,
    )
    return model
def build_vision_client(api_key,url):
    return OpenAI(
        api_key=api_key,
        base_url=url,
    )