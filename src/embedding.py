from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding(model_name):
    return HuggingFaceEmbeddings(model_name = model_name)