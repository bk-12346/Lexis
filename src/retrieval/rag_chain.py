from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

RAG_PROMPT = """You are a helpful assistant answering questions based strictly on the provided context.

RULES:
- Only use information from the provided context to answer
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer this question"
- Always cite which page(s) your answer comes from
- Be concise and precise

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def format_context(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string
    Each chunk is labelled with its source and page number
    This is what gets injected into the prompt
    """

    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "unknown")
        formatted.append(
            f"[Source {i+1} | File: {source} | Page: {page}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)

def get_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """
    We use gpt-4o-mini because:
    - Cheap — critical for production cost control
    - Fast — good for interactive applications
    - Smart enough for RAG (the context does most of the work)
    - Upgrade to gpt-4o only if quality is insufficient
    """
    return ChatOpenAI(model=model, temperature=0)

def ask(question: str, vectorstore: Chroma, llm: ChatOpenAI, k: int = 5) -> dict:
    """
    Full RAG Pipeline: retrieve + generate

    Returns a dict with:
    - answer: the LLM's response
    - sources: list of source documents used
    - context: the raw context fed to the LLM
    """

    # Step 1: Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    # Step 2: Format Context
    context = format_context(retrieved_docs)

    # Step 3: Build Prompt
    prompt = RAG_PROMPT.format(context=context, question=question)

    # Step 4: Generate Answer
    response = llm.invoke(prompt)

    # Step 5: Return answer with sources
    return{
        "answer": response.content,
        "sources": [
            {
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "excerpt": doc.page_content[:150]
            }
            for doc in retrieved_docs
        ]
    }

def print_response(response: dict) -> None:
    """Pretty print a RAG response with sources."""
    print("=" * 60)
    print("ANSWER:")
    print(response["answer"])
    print("\nSOURCES:")
    for i, source in enumerate(response["sources"]):
        print(f"  [{i+1}] {source['filename']} | Page {source['page']}")
        print(f"       \"{source['excerpt']}...\"")
    print("=" * 60)