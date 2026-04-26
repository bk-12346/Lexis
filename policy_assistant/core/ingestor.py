# policy specific ingestion with rich metadata

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

from test_ingestion import vectorstore

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
    for filename, registery_meta in DOCUMENT_REGISTRY.items():
        filepath = os.path.join(policy_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Loading: {filename}...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        # Enrich every page with registry metadata
        for doc in docs:
            doc.metadata.update({
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

def chunk_policy_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk policy docs into smaller chunks than research papers
    Policy docs have dense, precise language, smaller chunks preserve specific clauses better
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = int(i)
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks

def get_policy_vectorstore(embedder: OpenAIEmbeddings) -> Chroma:
    """
    Load or create the policy-specific vectorstore
    """
    return Chroma(
        persist_directory=POLICY_CHROMA_PATH,
        embedding_function=embedder,
        collection_name="policy_docs"
    )

def ingest_policy_documents(policy_dir: str, embedder: OpenAIEmbeddings) -> Chroma:
    """
    Full ingestion pipeline for policy docs
    Returns ready-to-query vectorstore
    """
    docs = load_policy_documents(policy_dir)
    print(f"\nTotal pages loaded: {len(docs)}")

    chunks = chunk_policy_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    vectorstore = get_policy_vectorstore(embedder)

    # Idempotency check
    ids = [
        f"{c.metadata['filename']}_{c.metadata['chunk_index']}"
        for c in chunks
    ]

    existing = vectorstore.get(ids=ids)
    existing_ids = set(existing["ids"])

    new_chunks = [c for c, id_ in zip(chunks, ids) if id_ not in existing_ids]
    new_ids = [id_ for id_ in ids if id_ not in existing_ids]

    if not new_chunks:
        print("All policy documents already ingested.")
    else:
        print(f"Adding {len(new_chunks)} new chunks...")
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)
        print("Ingestion complete.")
    
    return vectorstore
