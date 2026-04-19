from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

CHROMA_PATH = "chroma_db"

def get_vectorstore(embedder: OpenAIEmbeddings) -> Chroma:
    """
    Load existing vector store or create a new one
    Chroma goes the path CHROMA_PATH
    """
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedder
    )

def add_documents(vectorstore: Chroma, chunks: list[Document]) -> None:
    """
    Add chunks to the vector store with check that they are not already present there to save from duplication during ingestion

    Here we use source + chunk index as a simple unique index

    IN PROD: use document hashes as IDs to prevent duplication
    """

    # Build unique IDs for each chunk
    ids = [
        f"{chunk.metadata['filename']}_{chunk.metadata['chunk_index']}"
        for chunk in chunks
    ]

    # Check what's already in the store
    existing = vectorstore.get(ids=ids)
    existing_ids = set(existing["ids"])

    # Filter to only new chunks
    new_chunks = [
        chunk for chunk, id_ in zip(chunks, ids)
        if id_ not in existing_ids
    ]
    new_ids = [
        id_ for id_ in ids
        if id_ not in existing_ids
    ]

    if not new_chunks:
        print("All chunks already in vectorstore. Skipping...")
        return
    
    print(f"Adding {len(new_chunks)} new chunks to vector store...")
    vectorstore.add_documents(documents=new_chunks, ids=new_ids)
    
    print(f"DOne. Total chunks added: {len(new_chunks)}")

def similarity_search(vectorstore: Chroma, query: str, k: int = 5) -> list[Document]:
    """
    Retrieve top k most relevant chunks for a query
    k=5 is a sensible choice, enough context, not too much noise
    """

    return vectorstore.similarity_search(query, k=k)

def inspect_results(results: list[Document]) -> None:
    """Debug utility to inspect retrieval results."""
    print(f"Retrieved {len(results)} chunks\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('filename')} | Page: {doc.metadata.get('page')}")
        print(f"Content: {doc.page_content[:200]}...")
        print()