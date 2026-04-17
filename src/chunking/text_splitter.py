from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int =200
) -> list[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter
    
    This splitter splits text on paragraphs first, then sentences, then words.
    Only splits mid-word as a last resort
    This preserves the context better than a naive splitter that only splits on whitespace

    Args:
        documents: list of Document objects to spliit
        chunk_size: maximum number of characters in each chunk
        chunk_overlap: number of characters to overlap between chunks
    
    Returns:
        List of smaller Document objects, each with preserved metadata
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata - IMP for production
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i,
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    return chunks

def inspect_chunks(chunks: list[Document]) -> None:
    """
    Debugging utility to understand the chunking output
    """

    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    print(f"\nFirst chunk content:\n{chunks[0].page_content}")
    print(f"\nFirst chunk metadata: {chunks[0].metadata}")
    print(f"\nSecond chunk content preview:\n{chunks[1].page_content[:200]}")
    print(f"\n--- Overlap check ---")
    print(f"End of chunk 1: ...{chunks[0].page_content[-100:]}")
    print(f"Start of chunk 2: {chunks[1].page_content[:100]}...")