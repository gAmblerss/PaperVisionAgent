from utils import load_json,load_text
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from knowledge_loader import split_chunk

def add_vectorstore(text,info):
    chunks = split_chunk(text)
    docs = [Document(page_content=chunk,metadata=info) for chunk in chunks]
    return docs

def load_knowledge(json_path,embedding):
    """
    :param json_path: 知识库json
    :param embedding: embeding格式 目前是HuggingFace
    :return:
    """
    vectorstore = InMemoryVectorStore(embedding=embedding)
    json_data = load_json(json_path)
    for data in json_data:
        text = load_text(data['text'])
        vector = add_vectorstore(text,vectorstore)
        vectorstore.add_documents(vector)
    return vectorstore