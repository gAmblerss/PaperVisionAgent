"""
视觉智能体
"""
import logging

from dotenv import load_dotenv
from joblib.externals.loky.backend import context
from langchain_core.tools import retriever

from src import prompt, memory,model_loader,retriever
from .utils import encode_image_to_base64
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .logger_config import setup_logger

def build_pic_query(image_summary:str,user_question:str)->str:
    query = f"""
        图像摘要:
        {image_summary}
        用户问题：
        {user_question}
    """
    return query


def analyze_image(image_path:str,model_name:str,content:str,api_key:str,url:str)->str:
    client = model_loader.build_vision_client(api_key=api_key,url=url)
    image_64 = encode_image_to_base64(image_path=image_path)
    response = client.chat.completions.create( # 这里是走的底层SDK层 而model.invoke 是langchain封装后的，到后面也要走这层
        model = model_name,
        messages = [
            {
                "role":"system",
                "content":content
            },
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":"请分析这张图片，并按要求输出结构化摘要。"
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/jpeg;base64,{image_64}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


def answer_question(question,vectorstore,model,chat_history,k=8,fetch_k=12):
    related_doc = retriever.get_related_doc(vectorstore=vectorstore,k=k,fetch_k=fetch_k)

    context = retriever.doc_to_context(related_doc)
    system_prompt = prompt.System_prompt + "\n" + context

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    response = model.invoke(messages)
    chat_history = memory.upload_memory(question=question,answer=response.content,chat_history=chat_history)
    return response.content,related_doc,chat_history





def answer_with_rag(user_question:str,image_summary:str,vectorstore,model,chat_history,k,fetch_k):
    query = build_pic_query(image_summary=image_summary,user_question=user_question)
    related_doc = retriever.get_related_doc(vectorstore=vectorstore,question=query,k=k, fetch_k=fetch_k)
    context = retriever.doc_to_context(related_doc)

    system_prompt = f"""
        {prompt.ANSWER_SYSTEM_PROMPT}
        图像摘要：
        {image_summary}
        检测到的上下文:
        {context}
    """

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_question))

    response = model.invoke(messages)
    chat_history = memory.upload_memory(question=user_question,answer=response.content,chat_history=chat_history)
    return response.content,related_doc,chat_history

if __name__ == "__main__":
    import os
    from src import embedding,vectorstore,model_loader

    load_dotenv()
    setup_logger()
    logger = logging.getLogger(__name__)

    api_key = os.getenv("MOONSHOT_API_KEY")
    vision_api_key = os.getenv("MOONSHOT_VISION_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    chat_model_name = os.getenv("CHAT_MODEL_NAME")
    url = os.getenv("URL")
    k = os.getenv("K")
    fetch_k = os.getenv("FETCH_K")
    json_path = os.getenv("JSON_PATH")
    image_path = "../data/test1.jpg"

    # 图片理解
    image_summary = analyze_image(image_path=image_path,
                                        model_name=chat_model_name,
                                        content=prompt.VISION_SYSTEM_PROMPT,
                                        api_key=vision_api_key,
                                        url=url)
    logger.info(image_summary)

    # 图像文本对齐
    embed = embedding.HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = vectorstore.load_knowledge(json_path=json_path,embedding=embed)
    chat_model = model_loader.build_chat_model(model_name=chat_model_name,url=url)
    chat_history = []
    while True:
        question = input("\n请输入你的问题（输入quit退出）：")
        # 写成网页版就不用写这个了
        # image_path = input("\n请输入图像地址")
        # image_summary = analyze_image(image_path=image_path, model_name=chat_model, content=prompt.VISION_SYSTEM_PROMPT,
        #                               api_key=api_key, url=url)
        if question.lower() == "quit":
            break
        if question.lower() == "clear":
            chat_history = []
        if len(chat_history) == 12:
            chat_history = chat_history[2:]
        answer,doc,chat_history = answer_with_rag(user_question=question,
                                                  image_summary=image_summary,
                                                  vectorstore=vectorstore,
                                                  k=k,
                                                  fetch_k=fetch_k,
                                                  model=chat_model,
                                                  chat_history=chat_history
                                                  )

        try:
            print("\n检索到的内容：")

            for i, doc in enumerate(doc, 1):
                print(f"[{i}] {doc.page_content}{doc.metadata}")
        except Exception as e:
            print(e)

        logger.info("\n模型回答:")
        logger.info(answer)

        logger.info("\n日志写入！")
        logger.info(chat_history)





    # 文本对话
    # os.environ["MOONSHOT_API_KEY"] = api_key
    # embedding = embedding.HuggingFaceEmbeddings(model_name = embedding_model)
    # vectorstore = vectorstore.load_knowledge(json_path=json_path,embedding=embedding)
    # model = model_loader.build_chat_model(model_name=chat_model,url=url)
    # chat_history = []
    # while True:
    #     question = input("\n请输入你的问题(输入quit推出)：").strip()
    #     if question.lower() == "quit":
    #         break
    #     if question.lower() == "clear":
    #         chat_history = []
    #         print("记忆清空")
    #     if len(chat_history) == 12:
    #         chat_history = chat_history[2:]
    #     answer,docs,chat_history = answer_question(question,vectorstore,model,chat_history,k,fetch_k)
    #
    #     logger.info("\n检索到的内容:")
    #     for i,doc in enumerate(docs):
    #         logger.info(f"[{i}] {doc.page_content} {doc.metadata}")
    #     print("模型回答：")
    #     print(answer)
    #     logger.info("\n聊天记录写入日志")
    #     logger.info(chat_history)

