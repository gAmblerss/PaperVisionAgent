import streamlit as st


st.title("PaperVisionAgent")
st.header("论文图像智能体")

with st.form("login"):
    name = st.text_input("姓名")
    pwd = st.text_input("密码",type="password")
    submit = st.form_submit_button("login")
if submit:
    # todo: 密码校验，不同界面跳转
    pass
