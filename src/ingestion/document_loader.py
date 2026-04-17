from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

def load_single_PDF(file_path: str) -> list[Document]:
    """
    Load a single PDF file and return a list of Document objects
    Each page becomes one object with metadata attached to it
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Enrich metadata
    for doc in documents:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["filename"] = os.path.basename(file_path)
    
    return documents

def load_directory(dir_path: str, file_type: str = "pdf") -> list[Document]:
    """
    Load all the files of a given type from a directory
    Returns a list of Document objects
    This is how production systems work because they have multiple files to load
    """

    if file_type == "pdf":
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
    elif file_type == "txt":
        loader = DirectoryLoader(
            dir_path,
            glob = "**/*.text",
            loader_cls=TextLoader,
            show_progress=True
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = file_type
        doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
    
    return documents

def inspect_documents(documents : list[Document]) -> None:
    """
    Quick way to inspect the documents in the list; bug check
    In production, this is logged not printed
    """
    print(f"Loaded {len(documents)} documents")
    print(f"First document content length: {len(documents[0].page_content)}")
    print(f"First document metadata: {documents[0].metadata}")


