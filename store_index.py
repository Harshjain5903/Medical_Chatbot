from src.helper import load_pdf, chunk_splitter, load_embedding_model
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PC, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Ensure API key is found
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing! Add it to your .env file.")

# Initialize Pinecone correctly
pc = PC(api_key=PINECONE_API_KEY)

index_name = "test2"

# Check if the index exists
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Update based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )

index = pc.Index(index_name)

# Load and process data
data = load_pdf("data/")
chunks = chunk_splitter(data)
embeddings = load_embedding_model()

# Insert vectors into Pinecone
def insert_vectors(chunks, batch_size=50):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": chunk.page_content}
        })

        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            print(f"Inserted {len(vectors)} vectors into Pinecone.")
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)
        print(f"Inserted {len(vectors)} remaining vectors into Pinecone.")

insert_vectors(chunks, batch_size=50)
