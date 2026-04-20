from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from src.retrieval.rag_chain import format_context

@tool
def search_documents(query: str) -> str:
    """
    Search the document store for the relevant info to the query
    Use this when the user asks a specific question about the documents
    Returns relevant excerpts with page citations
    """
    # These will be injected when we initialize the agent
    from src.agent.agent import get_vectorstore_instance
    vectorstore = get_vectorstore_instance()

    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return "No relevant documents found for this query."

    return format_context(docs)

@tool
def summarize_topic(topic: str) -> str:
    """
    Retrieve and summarize all available information about a specific topic
    Use this when the user wants a comprehensive overview rather than a specific answer
    """
    from src.agent.agent import get_vectorstore_instance, get_llm_instance
    vectorstore = get_vectorstore_instance()
    llm = get_llm_instance()

    # Retrieve more chunks for summarization
    docs = vectorstore.similarity_search(topic, k=8)
    context = format_context(docs)

    prompt = f"""Based on the following context, provide a comprehensive summary of: {topic}
    
CONTEXT:
{context}

Provide a structured summary with key points. Cite page numbers."""
    
    response = llm.invoke(prompt)
    return response.content

@tool
def compare_concepts(concept_a: str, concept_b: str) -> str:
    """
    Compare two concepts or topics found in the documents.
    Use this when the user wants to understand differences or similarities between two things.
    """
    from src.agent.agent import get_vectorstore_instance, get_llm_instance
    vectorstore = get_vectorstore_instance()
    llm = get_llm_instance()
    
    docs_a = vectorstore.similarity_search(concept_a, k=4)
    docs_b = vectorstore.similarity_search(concept_b, k=4)
    
    context_a = format_context(docs_a)
    context_b = format_context(docs_b)
    
    prompt = f"""Compare and contrast the following two concepts based on the document context:

CONCEPT A - {concept_a}:
{context_a}

CONCEPT B - {concept_b}:
{context_b}

Provide a structured comparison highlighting similarities and differences. Cite page numbers."""
    
    response = llm.invoke(prompt)
    return response.content
