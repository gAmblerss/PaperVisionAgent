
from langchain.chat_models import init_chat_model
import os

def build_model(model_name,url):
    model = init_chat_model(
        # model_name = model_name,
        "kimi-k2.5",
        model_provider="openai",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url = url,
    )
    return model