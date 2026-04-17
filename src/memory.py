import streamlit as st
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



def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "image_path" not in st.session_state:
        st.session_state.image_path = ""
    if "image_summary" not in st.session_state:
        st.session_state.image_summary = ""
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = []
    if "last_answers" not in st.session_state:
        st.session_state.last_answers = ""

def add_to_history(user_question,ai_answer,maxmessages:int = 8):
    st.session_state.chat_history.append(HumanMessage(content = user_question))
    st.session_state.chat_history.append(AIMessage(content = ai_answer))

    if len(st.session_state.chat_history) > maxmessages:
        st.session_state.chat_history = st.session_state.chat_history[-maxmessages:]

def clear_memroy():
    st.session_state.chat_history = []
    st.session_state.image_path = None
    st.session_state.last_docs = []
    st.session_state.last_answers = ""
    st.session_state.image_summary = ""