import streamlit as st
from .pdf_handler import load_pdf
from .summarizer import summarize_pdf
from .rag_chain import qa_from_pdf

def build_ui():
    # Logo + Title inline
    col1, col2 = st.columns([1, 6])
    with col1:
       st.image("assets/logo.png", width=120)
    with col2:
       st.markdown("<h1 style='margin-top: 5px;'>QuestRAG: PDF QA and Summarizer Bot</h1>", unsafe_allow_html=True)
       st.markdown("---") 

    # File uploader and task options
    uploaded_pdf = st.file_uploader("📎 Upload your PDF", type=["pdf"])

    if uploaded_pdf:
        query = st.text_input("🔎 Ask a question:", "What is this paper about?")
        task = st.radio("🧠 Choose Task", ["Question Answering", "Summarization"])

        with st.spinner("⚙️ Processing..."):
            pages, temp_path = load_pdf(uploaded_pdf)

            if task == "Question Answering" and query:
                answer = qa_from_pdf(pages, query)
                st.subheader("💬 Answer:")
                st.write(answer)

            elif task == "Summarization":
                summary = summarize_pdf(pages)
                st.subheader("📝 Summary:")
                st.write(summary)