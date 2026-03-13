import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import PINECONE_API_KEY, EMBED_MODEL


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

INDEX_NAME="langgraph-rag-index"

def get_retriever():
    pass

def add_document_to_vectorstore(text_content:str):
    if not text_content:
        raise ValueError("Document cannot be empty")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    documents = text_splitter.create_documents([text_content])

    print(f"Splitting documents into {len(documents)} chunks for indexing")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    vectorstore.add_documents(documents)

    print(f'Successfully added {len(documents)} chunks to Pinecone index {INDEX_NAME}.')