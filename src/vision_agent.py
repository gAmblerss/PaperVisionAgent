"""
视觉智能体
"""

import logging
from src import prompt, memory
from langchain_core.messages import SystemMessage,HumanMessage
from logger_config import setup_logger




def answer_question(question,vectorstore,model,chat_history):
    retriever = vectorstore.as_retriever(search_kwags={"k":5})
    related_doc = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in related_doc])
    system_prompt = prompt.System_prompt + "\n" + context

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    response = model.invoke(messages)
    chat_history = memory.upload_memory(question=question,answer=response.content,chat_history=chat_history)
    return response.content,related_doc,chat_history

if __name__ == "__main__":
    import os
    from src import embedding,vectorstore,model_loader

    setup_logger()
    logger = logging.getLogger(__name__)

    os.environ["MOONSHOT_API_KEY"] = "sk-fl2pwY0oUz58Q3clirsNfjlwEbhmxtJErnzSYnbE1FzypW11"
    json_path = "../data/manifest.json"
    embedding = embedding.HuggingFaceEmbeddings(model_name = "BAAI/bge-small-zh-v1.5")
    vectorstore = vectorstore.load_knowledge(json_path=json_path,embedding=embedding)
    model = model_loader.build_model(model_name= "kimi-k2.5",url="https://api.moonshot.cn/v1")
    chat_history = []
    while True:
        question = input("\n请输入你的问题(输入quit推出)：").strip()
        if question.lower() == "quit":
            break
        if question.lower() == "clear":
            chat_history = []
            print("记忆清空")
        if len(chat_history) == 12:
            chat_history = chat_history[2:]
        answer,docs,chat_history = answer_question(question,vectorstore,model,chat_history)

        logger.info("\n检索到的内容:")
        for i,doc in enumerate(docs):
            logger.info(f"[{i}] {doc.page_content} {doc.metadata}")
        print("模型回答：")
        print(answer)
        logger.info("\n聊天记录写入日志")
        logger.info(chat_history)

