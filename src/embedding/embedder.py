from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

def get_embedder(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Returns an embedding model instance.

    We use text-embedding-3-small because:
    - it is a smaller but good model
    - will make 1536 dim vectors which is good for RAG
    - good balance of speed, cost and quality

    In production, we configure this using an environment variable so that we can switch 
    between models without having to change the code.
    """

    return OpenAIEmbeddings(
        model = model,
        api_key= os.getenv("OPENAI_API_KEY")
    )

def embed_text(text: str, embedder: OpenAIEmbeddings) -> list[float]:
    """
    Embed a single string
    Used for embedding user queries during retrieval time
    Model must be same as the one used in embedder
    """

    return embedder.embed_query(text)

def inspect_embedding(text: str, embedder: OpenAIEmbeddings) -> None:
    """Debug utility to understand what an embedding looks like."""

    vector = embed_text(text, embedder)

    print(f"Text: {text}")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print(f"Last 5 values: {vector[-5:]}")