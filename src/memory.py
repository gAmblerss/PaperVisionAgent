from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


def upload_memory(question,answer,chat_history):
    """
    :param system_prompt:
    :param question:
    :param answer:
    :param chat_history:
    :return:
    """

    chat_history.append(HumanMessage(content = question))
    chat_history.append(AIMessage(content = answer))

    return chat_history
