import os
import glob
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
import streamlit as st
print(os.getenv("HF_API_TOKEN"))
HF_API_TOKEN = "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"
LLAMA_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

UPLOAD_FOLDER = "uploaded_documents"

def save_uploaded_files(uploaded_files):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    for uploaded_file in uploaded_files:
        with open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

def load_documents(folder_path):
    docs = []
    for filepath in glob.glob(os.path.join(folder_path, "*")):
        ext = os.path.splitext(filepath)[-1].lower()
        if ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyMuPDFLoader(filepath)
        else:
            continue
        loaded = loader.load()
        docs.extend(loaded)
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedding_function)
    return vectorstore

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_TOKEN,
)

def create_llm():
    return HuggingFaceEndpoint(
        endpoint_url=LLAMA_MODEL_URL,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.2,
        top_k=50,
        server_kwargs={
            "headers": {"Authorization": f"Bearer {HF_API_TOKEN}"}
        },
    )

def create_reranker_retriever(vectorstore):
    reranker_model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME
    )
    reranker = CrossEncoderReranker(
        model=reranker_model,
        top_n=5
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    return reranker_retriever

def create_rag_chain(reranker_retriever):
    llm = create_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=reranker_retriever,
        chain_type="stuff",
        return_source_documents=False
    )
    return qa_chain

import re

def clean_answer(answer):
    # Stronger regex-based cleanup
    garbage_patterns = [
        r"(?i)If you want to provide.*?sufficient\.",  # Remove phrases suggesting more details
        r"(?i)Note:.*?extraction.*?sufficient\.",     # Remove "Note: The answer is a direct extraction"
        r"(?i)Let's try another question.*",           # Remove "Let's try another question"
        r"(?i)Helpful Answer:.*",                     # Remove "Helpful Answer:"
        r"(?i)If you don't know the answer.*",         # Remove "If you don't know the answer"
        r"\s{2,}",  # Remove multiple spaces
        r"^\s+|\s+?$",  # Remove leading/trailing whitespace
    ]
    
    # Apply each pattern to the answer
    for pattern in garbage_patterns:
        answer = re.sub(pattern, "", answer)
        
    # Clean up any residual spaces and newline characters
    answer = answer

    return answer



def run_rag(folder_path, question):
    documents = load_documents(folder_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    reranker_retriever = create_reranker_retriever(vectorstore)
    rag_chain = create_rag_chain(reranker_retriever)
    formatted_question = (
        "[INST] "
        "Answer the following question based on the documents uploaded. "
        "Do not add any notes, disclaimers, or extra information. "
        "If you don't know the answer, try to find and if not found say 'I don't know'. "
        "Question: " + question +
        " [/INST]"
    )
    
    answer = rag_chain.invoke({"query": formatted_question})
    cleaned_answer = clean_answer(answer["result"])
    cleaned_answer = re.sub(r"Note:.*?If you want to provide.*", "", cleaned_answer, flags=re.DOTALL)
    return cleaned_answer


def main():
    st.title("Document Question Answering System 📄🔍")
    
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if uploaded_files and question:
            save_uploaded_files(uploaded_files)
            with st.spinner("Processing..."):
                final_answer = run_rag(UPLOAD_FOLDER, question)
                st.subheader("Answer:")
                st.write(final_answer)
        else:
            st.error("Please upload documents and enter a question.")

if __name__ == "__main__":
    main()
