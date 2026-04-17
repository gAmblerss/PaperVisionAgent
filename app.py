import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.memory import init_session_state,add_to_history,clear_memroy
from src.vision_agent import analyze_image
from src.model_loader import build_chat_model
from src.vectorstore import load_knowledge
from src.rag_pipeline import answer_with_rag
from src.embedding import get_embedding
from src.prompt import VISION_SYSTEM_PROMPT
load_dotenv()



json_path = os.getenv("JSON_PATH")
embed_model_name = os.getenv("EMBEDDING_MODEL_NAME")
chat_model_name = os.getenv("CHAT_MODEL_NAME")
chat_api_key = os.getenv("MOONSHOT_API_KEY")
url = os.getenv("URL")
vision_api_key = os.getenv("MOONSHOT_VISION_API_KEY")



st.set_page_config(
    page_title="PaperVisionAgent",
    layout="wide",
)

st.title("PaperVisionAgent")

init_session_state()



col1, col2 = st.columns([3,1])
with col1:
    uploaded_file = st.file_uploader(
        "上传论文截图、模型结构图、实验图或表格",
        type=['png','jpg','jpeg'],
    )
with col2:
    if st.button("清空记忆"):
        clear_memroy()
        st.success("记忆已清空")

if uploaded_file is not None:
    save_dir = "data/uploads"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,uploaded_file.name)
    with open(save_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    st.session_state.image_path = save_path
    image = Image.open(save_path)
    st.image(image,caption="当前图片",use_column_width=True)
    if st.button("生成图像摘要"):
        with st.spinner("正在分析图片...."):
            summary = analyze_image(
                image_path = save_path,
                model_name = chat_model_name,
                content = VISION_SYSTEM_PROMPT,
                api_key = vision_api_key,
                url = url,
            )
        st.session_state.image_summary = summary

if st.session_state.image_summary:
    st.subheader("当前图像摘要：")
    st.write(st.session_state.image_summary)

if "vectorstore" not in st.session_state:
    embed = get_embedding(embed_model_name)
    st.session_state.vectorstore = load_knowledge(json_path=json_path,embedding=embed)
if "model" not in st.session_state:
    st.session_state.model = build_chat_model(
        model_name=chat_model_name,
        url=url,
    )
question = st.chat_input("请输入你的问题")

if question:
    if not st.session_state.image_summary:
        st.warning("请先上传图片并生成图像摘要")
    else:
        with st.spinner("正在检索知识库并生成答案"):
            answer,docs = answer_with_rag(
                user_question=question,
                image_summary=st.session_state.image_summary,
                vectorstore=st.session_state.vectorstore,
                model=st.session_state.model,
                chat_history=st.session_state.chat_history,
                k = os.getenv("K"),
                fetch_k=os.getenv("FETCH_K"),
            )
            st.session_state.last_docs = docs
            st.session_state.last_answers = answer
            add_to_history(user_question=question,ai_answer=answer,maxmessages=8)

for msg in st.session_state.chat_history:
    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)
if st.session_state.last_answers:
    st.subheader("最后一次回答")
    st.write(st.session_state.last_answers)
if st.session_state.last_docs:
    st.subheader("最近一次命中的知识库片段")
    for i,doc in enumerate(st.session_state.last_docs,1):
        st.markdown(f"**[{i}] 来源：{doc.metadata}**")
        st.write(doc.page_content)