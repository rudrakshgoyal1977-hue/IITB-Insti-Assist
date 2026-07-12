# IITB-Insti-Assist
A RAG-Powered AI Assistant for IIT Bombay

# IITB Insti-Assist — RAG-Powered AI Assistant

This repository contains the source code for IITB Insti-Assist, a Retrieval-Augmented Generation (RAG) assistant designed to answer questions specifically about IIT Bombay. This project focuses on the **Academic Assistant** scope, handling queries about course registration, grading policies, the academic calendar, and exam rules.

## Features
* **Grounded Generation:** The model is strictly instructed to say "I don't know" if the answer is not in the source documents, preventing hallucinations.
* **Source Transparency:** The UI explicitly displays the retrieved document chunks used to generate the answer.
* **Local Vector Search:** Utilizes FAISS and `all-MiniLM-L6-v2` embeddings for fast, accurate context retrieval.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rudrakshgoyal1977-hue/IITB-Insti-Assist]
   cd IITB-Insti-Assist
2. **Install dependencies**
   Run: pip install -r requirements.txt
3. **Add your API Key**
   Open app.py and replace "YOUR_GEMINI_API_KEY" with a valid Google Gemini API key.
4. **Run the application**
   Run: streamlit run app.py
