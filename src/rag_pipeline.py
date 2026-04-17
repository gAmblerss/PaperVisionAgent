from src import prompt, memory,model_loader,retriever
from langchain_core.messages import SystemMessage,HumanMessage


def build_pic_query(image_summary:str,user_question:str)->str:
    query = f"""
        图像摘要:
        {image_summary}
        用户问题：
        {user_question}
    """
    return query
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
    # chat_history = memory.upload_memory(question=user_question,answer=response.content,chat_history=chat_history)
    return response.content,related_doc