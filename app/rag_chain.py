from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import os
import tempfile
import re
import streamlit as st

# Ensure consistent model paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(ROOT_DIR, "..", "saved_model", "flan-t5-base"))
EMBEDDING_DIR = os.path.normpath(os.path.join(ROOT_DIR, "..", "saved_model", "all-MiniLM-L6-v2"))

# Clean repetitive answers
import re

def clean_answer(text):
    if not text.strip():
        return "⚠️ No answer could be extracted from the document."

    # Remove LaTeX math expressions
    text = re.sub(r'\$.*?\$', '', text)

    # Remove repeated sentences
    sentences = text.strip().split('. ')
    seen = set()
    unique_sentences = [s.strip() for s in sentences if s and s not in seen and not seen.add(s)]

    # Join cleaned sentences
    cleaned_text = '. '.join(unique_sentences).strip()

    # Capitalize first letter
    if cleaned_text and cleaned_text[0].islower():
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]

    # Add period at the end if missing
    if not cleaned_text.endswith('.'):
        cleaned_text += '.'

    return f"Answer: {cleaned_text}"

@st.cache_resource(show_spinner="🔄 Loading models and building vector store...")
def load_models_and_vectorstore(_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(_pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_DIR)

    chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_db")

    # Clear old Chroma cache
    if os.path.exists(chroma_dir):
       import shutil
       shutil.rmtree(chroma_dir)

    vector_db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_dir)
    retriever = vector_db.as_retriever()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return retriever, llm


def qa_from_pdf(pages, query):
    retriever, llm = load_models_and_vectorstore(pages)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    raw_answer = qa_chain.run(query)
    return clean_answer(raw_answer)