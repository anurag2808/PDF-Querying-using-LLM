import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os

# Set up API key
GOOGLE_API_KEY = "AIzaSyCzR6La9YT8Ww51fyhGiGTTQQcqzia31Z4"  # Replace with your API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit UI
st.title("📄 PDF Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load and process PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.from_documents(texts, embeddings)

    # Create RetrievalQA chain
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest"), retriever=retriever)

    # Chat UI
    st.header("Ask questions about the PDF")
    query = st.text_input("Enter your question:")
    
    if query:
        response = qa({"query": query})
        st.write("**Answer:**", response["result"])

