"""
封装检索逻辑
"""

def get_related_doc(vectorstore,question:str,k,fetch_k)->str:
    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwags = {'k':k,fetch_k:fetch_k},
    )
    related_docs = retriever.invoke(question)
    return related_docs

def doc_to_context(related_docs)->str:
    return "\n".join([doc.page_content for doc in related_docs])