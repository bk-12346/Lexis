import sys
sys.path.append('.')

from src.ingestion.document_loader import load_single_PDF, load_directory, inspect_documents

docs = load_single_PDF("data/raw/Towards_an_ai_coscientist.pdf")
inspect_documents(docs)