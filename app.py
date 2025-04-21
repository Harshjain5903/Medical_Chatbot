# --- Start of app.py ---yo yo honey singh

from flask import Flask, render_template, jsonify, request
from src.helper import load_embedding_model
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *  # Assuming prompt_template is defined here
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Load environment variables
log.info("Loading environment variables...")
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    log.error("PINECONE_API_KEY environment variable is not set or is empty!")
    raise ValueError("PINECONE_API_KEY environment variable is not set or is empty")
log.info("PINECONE_API_KEY loaded successfully.")

# Configuration constants
PINECONE_INDEX_NAME = "test2"
EXPECTED_EMBEDDING_DIMENSION = 384
PINECONE_METRIC = 'cosine'
PINECONE_CLOUD = 'aws'
PINECONE_REGION = 'us-east-1'
LLM_MODEL_PATH = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
LLM_MODEL_TYPE = "llama"
LLM_TEMPERATURE = 0.8
LLM_MAX_NEW_TOKENS = 64
RETRIEVER_K = 1

# Initialize Pinecone
log.info("Initializing Pinecone client...")
try:
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    log.info("Pinecone client initialized.")
except Exception as e:
    log.error(f"Failed to initialize Pinecone client: {e}", exc_info=True)
    raise

# Check or create index
try:
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        log.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Creating new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EXPECTED_EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        log.info(f"Index '{PINECONE_INDEX_NAME}' created. It's currently empty.")
    else:
        index_desc = pc.describe_index(PINECONE_INDEX_NAME)
        if index_desc.dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(f"Dimension mismatch in index '{PINECONE_INDEX_NAME}'")
        log.info(f"Index '{PINECONE_INDEX_NAME}' exists with correct dimension.")
    index = pc.Index(PINECONE_INDEX_NAME)
    log.info(f"Connected to index '{PINECONE_INDEX_NAME}'. Stats: {index.describe_index_stats()}")
except Exception as e:
    log.error(f"Pinecone index error: {e}", exc_info=True)
    raise

# Load embeddings
log.info("Loading embedding model...")
try:
    embeddings = load_embedding_model()
    log.info("Embedding model loaded.")
except Exception as e:
    log.error(f"Error loading embedding model: {e}", exc_info=True)
    raise

# Setup vector store
try:
    docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    log.info("PineconeVectorStore initialized.")
except Exception as e:
    log.error(f"Failed to initialize PineconeVectorStore: {e}", exc_info=True)
    raise

# Setup prompt
try:
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    log.info("Prompt template ready.")
except Exception as e:
    log.error(f"Prompt setup error: {e}", exc_info=True)
    raise

# Load LLM
if not os.path.exists(LLM_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {LLM_MODEL_PATH}")
try:
    llm = CTransformers(
        model=LLM_MODEL_PATH,
        model_type=LLM_MODEL_TYPE,
        config={'max_new_tokens': LLM_MAX_NEW_TOKENS, 'temperature': LLM_TEMPERATURE}
    )
    log.info("LLM loaded successfully.")
except Exception as e:
    log.error(f"LLM loading error: {e}", exc_info=True)
    raise

# Setup RetrievalQA chain
try:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': RETRIEVER_K}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    log.info("RetrievalQA chain setup complete.")
except Exception as e:
    log.error(f"QA chain setup error: {e}", exc_info=True)
    raise

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        log.info(f"Received: {msg}")
        result = qa.invoke({"query": msg})
        response_text = result.get("result", "Sorry, I couldn't find an answer.").strip()
        return jsonify({"response": response_text})
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"response": f"Error: {type(e).__name__}"}), 500

# Run the server
if __name__ == '__main__':
    log.info("Starting Flask server on 0.0.0.0:8080...")
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)

# --- End of app.py ---