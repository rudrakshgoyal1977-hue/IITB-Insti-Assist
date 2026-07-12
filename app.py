import os
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Google Gemini API (Replace with your own API key)
# Alternatively, you can use OpenAI or Claude API as suggested in the requirements.
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- CORE RAG FUNCTIONS ---

@st.cache_resource
def setup_rag_pipeline(data_dir="data"):
    """
    Ingests 5+ documents, chunks them, and builds a FAISS vector search index.
    """
    documents = []
    # 1. Data Ingestion: Load all PDFs from the 'data' directory
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
    
    # 2. Chunking: Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embedding & Vector Search: Create FAISS index using sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    return vector_db

def generate_grounded_response(query, vector_db):
    """
    Retrieves context and forces the LLM to answer ONLY from the context.
    """
    # Retrieve top-k relevant chunks
    docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([f"Source: {doc.metadata['source']} \nContent: {doc.page_content}" for doc in docs])
    
    # 4 & 5. LLM Integration & Grounded Answering
    prompt = f"""
    You are IITB Insti-Assist, an AI specialist for IIT Bombay. 
    Answer the user's question using ONLY the provided context. 
    If the answer is not explicitly contained in the context, you MUST reply with exactly: "I don't know." Do not guess or hallucinate.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return response.text, docs

# --- STREAMLIT WEB INTERFACE ---

st.title("🎓 IITB Insti-Assist: Academic Guide")
st.markdown("A RAG-Powered AI Assistant for IIT Bombay. Ask me about course registration, grading policies, and exam rules!")

# Initialize Database
try:
    vector_db = setup_rag_pipeline("data")
    st.sidebar.success("Database loaded and indexed successfully.")
except Exception as e:
    st.sidebar.error("Error loading data. Ensure you have a 'data' folder with 5+ PDFs.")

# User Input
user_query = st.text_input("Ask a question about IITB Academics:")

if st.button("Search"):
    if user_query:
        with st.spinner("Searching institute documents..."):
            answer, sources = generate_grounded_response(user_query, vector_db)
            
            st.subheader("Answer")
            st.write(answer)
            
            # 6. Source display
            if answer.strip().lower() != "i don't know.":
                st.subheader("📚 Sources Used")
                for i, doc in enumerate(sources):
                    with st.expander(f"Source {i+1}: {os.path.basename(doc.metadata['source'])}"):
                        st.write(doc.page_content)
    else:
        st.warning("Please enter a question.")