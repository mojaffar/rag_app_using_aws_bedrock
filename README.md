📄 Chat with PDF using AWS Bedrock (RAG Application)

A Retrieval-Augmented Generation (RAG) based application that allows users to upload PDFs and ask questions from them using powerful LLMs like Claude and Llama3 via AWS Bedrock.

🚀 Features

📄 Chat with multiple PDF documents

🔍 Semantic search using embeddings

🤖 Multiple LLM support:

Claude Haiku (Fast)

Claude Sonnet (Advanced)

Llama3

⚡ Fast retrieval using FAISS vector store

🧠 Context-aware responses (RAG pipeline)

🌐 Simple UI using Streamlit

🏗️ Architecture

User Input
   ↓
Streamlit UI
   ↓
FAISS Vector Store (Retrieval)
   ↓
AWS Bedrock (Embeddings + LLM)
   ↓
Generated Answer

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python

LLM Provider: AWS Bedrock

Embeddings: Amazon Titan Embeddings

Vector Store: FAISS

Framework: LangChain

⚙️ Setup Instructions

1. Clone the Repository

git clone <your-repo-url>

cd <project-folder>

2. Install Dependencies

pip install -r requirements.txt

3. Configure AWS

Make sure you have AWS credentials configured:

aws configure

Provide:

Access Key

Secret Key

Region (e.g., us-east-1)

4. Run the Application

streamlit run app.py

📌 Usage

Upload PDFs to the data/

Click "Create / Update Vectors"

Ask a question in the input box

Select the model

Click "Get Answer"