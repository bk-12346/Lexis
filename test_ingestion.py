import sys
sys.path.append('.')

from src.ingestion.document_loader import load_single_PDF, load_directory, inspect_documents
from src.chunking.text_splitter import chunk_documents, inspect_chunks

docs = load_single_PDF("data/raw/Towards_an_ai_coscientist.pdf")
print(f"Pages loaded: {len(docs)}\n")

chunks = chunk_documents(docs)
inspect_chunks(chunks)