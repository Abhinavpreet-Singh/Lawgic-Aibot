import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Ensure non-null text
        except Exception as e:
            print(f"Error reading {pdf.name}: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to store text in FAISS vector database
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up a conversational AI model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "answer is not available in the context."
    Don't provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    with st.spinner("Analyzing documents..."):
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

    st.markdown(
        f"""
        <div class="response-box">
            <h3>Response:</h3>
            <p>{response["output_text"]}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Function to add logo
def add_logo():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px; padding: 20px 0; border-bottom: 1px solid #a7967e;">
            <h1 style="font-size: 42px; font-weight: 800; color: #251c1a; letter-spacing: 2px; margin-bottom: 0;">
                <span style="color: #826f5a;">L</span>AWGIC
            </h1>
            <p style="font-size: 14px; color: #826f5a; letter-spacing: 1px; margin-top: 0;">Legal Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI setup
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Lawgic | Chat with Legal Documents",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Apply custom CSS with direct styling
    st.markdown(
        """
        <style>
        /* Base styles */
        body {
            background-color: #f3eee5;
            color: #251c1a;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Header styling */
        h1 {
            color: #251c1a;
            font-weight: 700;
            font-size: 32px;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }
        
        h1:after {
            content: "";
            display: block;
            width: 80px;
            height: 3px;
            background-color: #826f5a;
            margin: 15px auto 0;
        }
        
        h2, h3 {
            color: #251c1a;
            font-weight: 600;
        }
        
        /* Main container */
        .main-container {
            background-color: #f3eee5;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(37, 28, 26, 0.1);
            margin-bottom: 30px;
            border: 1px solid rgba(37, 28, 26, 0.08);
        }
        
        /* Input field styling */
        .stTextInput > div > div > input {
            background-color: #ffffff;
            color: #251c1a;
            border: 2px solid #a7967e;
            padding: 16px;
            font-size: 16px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(37, 28, 26, 0.05);
            transition: all 0.3s ease;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #251c1a;
            color: #f3eee5;
            border: none;
            padding: 14px 24px;
            font-weight: 600;
            font-size: 16px;
            letter-spacing: 0.5px;
            border-radius: 10px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(37, 28, 26, 0.1);
        }
        
        .stButton > button:hover {
            background-color: #826f5a;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 28, 26, 0.1);
        }
        
        /* Response box */
        .response-box {
            background-color: #ffffff;
            border-left: 6px solid #826f5a;
            padding: 25px;
            border-radius: 12px;
            margin-top: 25px;
            box-shadow: 0 6px 12px rgba(37, 28, 26, 0.1);
        }
        
        .response-box h3 {
            color: #826f5a;
            margin-top: 0;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 15px;
            border-bottom: 1px solid #c8b28e;
            padding-bottom: 10px;
        }
        
        .response-box p {
            font-size: 16px;
            line-height: 1.6;
            color: #251c1a;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #251c1a !important;
        }
        
        [data-testid="stSidebarNav"] {
            background-color: #251c1a !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdown"] {
            color: #f3eee5;
        }
        
        /* Card section */
        .card-section {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(37, 28, 26, 0.1);
            margin-bottom: 24px;
        }
        
        /* Placeholder text */
        .placeholder-text {
            color: #a7967e;
            font-style: italic;
            text-align: center;
            padding: 30px 0;
        }
        
        /* Custom sidebar sections */
        .sidebar-section {
            background-color: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        
        /* Success message */
        .success-message {
            background-color: rgba(76, 175, 80, 0.1);
            color: #4CAF50;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            margin: 15px 0;
        }
        
        /* Error message */
        .error-message {
            background-color: rgba(244, 67, 54, 0.1);
            color: #F44336;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #F44336;
            margin: 15px 0;
        }
        
        /* Upload indicator */
        .upload-indicator {
            display: flex;
            align-items: center;
            margin-top: 10px;
            color: #f3eee5;
            font-size: 14px;
        }
        
        .upload-indicator .dot {
            width: 10px;
            height: 10px;
            background-color: #c8b28e;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        /* Force sidebar background */
        div[data-testid="stSidebarUserContent"] {
            background-color: #251c1a;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Layout with columns
    col1, col2 = st.columns([1, 5])
    
    with col2:
        add_logo()
        st.markdown("<h1>Chat with Legal Documents</h1>", unsafe_allow_html=True)
        
        # Main content container
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="card-section">', unsafe_allow_html=True)
        user_question = st.text_input("📝 Enter your legal question here...", 
                                     placeholder="e.g., What are the key points in this judgment?")
        
        if user_question:
            user_input(user_question)
        else:
            st.markdown('<div class="placeholder-text">Your conversation will appear here after you ask a question.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #f3eee5; text-align: center; padding: 10px 0; font-size: 24px;'>Lawgic Control Panel</h2>", unsafe_allow_html=True)
        st.markdown('<div style="background-color: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 12px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #f3eee5; font-size: 18px; margin-bottom: 15px;'>Upload Documents</h3>", unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Select PDF files to analyze", 
            accept_multiple_files=True,
            type="pdf"
        )
        
        if pdf_docs:
            st.markdown('<div style="display: flex; align-items: center; margin-top: 10px; color: #f3eee5; font-size: 14px;"><div style="width: 10px; height: 10px; background-color: #c8b28e; border-radius: 50%; margin-right: 8px;"></div>Files ready for processing</div>', unsafe_allow_html=True)
        
        process_btn = st.button("Process Documents", key="process_btn")
        
        if process_btn:
            if not pdf_docs:
                st.markdown('<div style="background-color: rgba(244, 67, 54, 0.1); color: #F44336; padding: 12px; border-radius: 8px; border-left: 4px solid #F44336; margin: 15px 0;">Please upload at least one document.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.markdown('<div style="background-color: rgba(244, 67, 54, 0.1); color: #F44336; padding: 12px; border-radius: 8px; border-left: 4px solid #F44336; margin: 15px 0;">No text could be extracted from the uploaded documents.</div>', unsafe_allow_html=True)
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.markdown('<div style="background-color: rgba(76, 175, 80, 0.1); color: #4CAF50; padding: 12px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0;">✅ Documents processed successfully! You can now ask questions.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats or info section
        st.markdown('<div style="background-color: rgba(255, 255, 255, 0.08); padding: 20px; border-radius: 12px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #f3eee5; text-align: center; font-size: 18px;'>About Lawgic</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #f3eee5; font-size: 14px; line-height: 1.6;'>Powered by advanced AI to help legal professionals analyze documents efficiently and extract valuable insights.</p>", unsafe_allow_html=True)
        
        # Add a simple usage guide
        st.markdown("<h4 style='color: #c8b28e; margin-top: 20px; font-size: 16px;'>How to use</h4>", unsafe_allow_html=True)
        st.markdown("""
        <ol style='color: #f3eee5; font-size: 14px; padding-left: 20px;'>
            <li>Upload legal documents (PDF format)</li>
            <li>Click "Process Documents" button</li>
            <li>Ask questions about the content</li>
        </ol>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Ensure correct execution
if __name__ == "__main__":
    main()
