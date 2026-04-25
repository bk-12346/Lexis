# policy specific ingestion with rich metadata

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Separate vectore store for the policy docs - different from the RAG pipeline
POLICY_CHROMA_PATH = "policy_chroma_db"

# DOCUMENT REGISTRY - metadat we know in advance for each doc
DOCUMENT_REGISTRY = {
    "nist_ai_rmf.pdf": {
        "title" : "NIST AI Risk Management Framework",
        "category" : "AI Governance",
        "issuer" : "NIST",
        "year" : 2023,
        "clearance_level" : 1
    },
    "us_executive_order_ai_2023.pdf" : {
        "title" : "US Executive Order on AI",
        "category" : "AI Policy",
        "issuer" : "White House",
        "year" : 2023,
        "clearance_level" : 1
    },
    "unesco_ai_ethics.pdf" : {
        "title" : "UNESCO AI Ethics Recommendation",
        "category" : "AI Ethics",
        "issuer" : "UNESCO",
        "year" : 2021,
        "clearance_level" : 1
    }
}

def load_policy_documents(policy_dir: str) -> list[Document]:
    """
    Load all the policy docs with rich metadata from registry.
    This is especially for enterprise systems:
    metadata is predefined, not just extracted from the file
    """
    all_docs = []
    for filename, registery_meta in DOCUMENT_REGISTRY.items:
        filepath = os.path.join(policy_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Loading: {filename}...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        # Enrich every page with registry metadata
        for doc in docs:
            doc.meta.update({
                "filename" : filename,
                "title" : registery_meta["title"],
                "category" : registery_meta["category"],
                "issuer" : registery_meta["issuer"],
                "year" : registery_meta["year"],
                "clearance_level" : registery_meta["clearance_level"],
                "source_type" : "policy"
            })
        
        all_docs.extend(docs)
        print(f" Loaded {len(docs)} pages")
    
    return all_docs



