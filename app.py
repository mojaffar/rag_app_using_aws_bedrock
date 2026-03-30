import boto3
import streamlit as st

# Embeddings & LLM
from langchain_aws import ChatBedrock, BedrockEmbeddings

# Data Ingestion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Store
from langchain_community.vectorstores import FAISS

# Prompt & RAG
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------------- Bedrock Client -------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

# ------------------- Data Ingestion -------------------
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF files found in 'data' folder")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)
    return docs

# ------------------- Vector Store -------------------
def get_vector_store(docs):
    if not docs:
        raise ValueError("No documents to process!")

    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# ------------------- LLMs -------------------
def get_claude_haiku():
    return ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.5
        }
    )

def get_claude_sonnet():
    return ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.5
        }
    )

def get_llama3():
    return ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={
            "max_gen_len": 512,
            "temperature": 0.5
        }
    )

# ------------------- Prompt -------------------
prompt_template = """
Human: Use the following context to answer the question.
Provide a detailed answer of at least 250 words.
If you don't know the answer, just say that you don't know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ------------------- RAG Pipeline -------------------
def get_response_llm(llm, vectorstore_faiss, query):

    retriever = vectorstore_faiss.as_retriever(
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    return response

# ------------------- Streamlit App -------------------
def main():
    st.set_page_config(page_title="Chat PDF")

    st.header("📄 Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a question from your PDFs")

    # 🔥 Model Selection
    model_option = st.selectbox(
        "Choose Model",
        ["Claude Haiku (Fast)", "Claude Sonnet (Best)", "Llama3"]
    )

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Vector Store")

        if st.button("Create / Update Vectors"):
            with st.spinner("Processing PDFs..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("✅ Vector store created!")

    # Get Answer
    if st.button("Get Answer"):

        if not user_question:
            st.warning("Please enter a question")
            return

        with st.spinner("Thinking..."):

            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )

            # 🔥 Dynamic model selection
            if model_option == "Claude Haiku (Fast)":
                llm = get_claude_haiku()

            elif model_option == "Claude Sonnet (Best)":
                llm = get_claude_sonnet()

            elif model_option == "Llama3":
                llm = get_llama3()

            answer = get_response_llm(llm, faiss_index, user_question)

            st.write(answer)
            st.success("✅ Done")

if __name__ == "__main__":
    main()