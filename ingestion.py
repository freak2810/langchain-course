from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        "/Users/adityamanikanthrao/Developer/Web Dev/Courses/Udemy/Learn AI Engineering/mediumblog1.txt"
    )
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text")

    embeddings = PineconeEmbeddings(model="llama-text-embed-v2")

    # pc = Pinecone()  # reads PINECONE_API_KEY
    # pc.create_index(
    #     name=os.environ.get("INDEX_NAME"),
    #     dimension=1024,
    #     metric="cosine",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME")
    )
