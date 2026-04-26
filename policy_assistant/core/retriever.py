from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

RAG_PROMPT = """You are a policy document assistant for government and enterprise clients.
Answer questions strictly based on the provided policy document context.

RULES:
- Only use information from the provided context
- If the context does not contain enough information, say so explicitly
- Always cite the document title, issuer, and page number
- Be precise — policy language matters, do not paraphrase loosely
- If multiple documents address the question, synthesize their positions

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def format_policy_context(docs: list[Document]) -> str:
    """
    Format retrieved chunks with full policy metadata
    """
    formatted = []

    for i, doc in enumerate(docs):
        formatted.append(
            f"[Source {i+1}]\n"
            f"Document: {doc.metadata.get('title', 'Unknown')}\n"
            f"Issuer: {doc.metadata.get('issuer', 'Unknown')}\n"
            f"Year: {doc.metadata.get('year', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"Category: {doc.metadata.get('category', 'Unknown')}\n\n"
            f"{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(formatted)

def compute_confidence(docs: list[Document], query: str) -> float:
    """
    Rough confidence score based on retrieval

    IN PRODUCTION: Use cross-encoder reranking scores

    Here, we use a simple heuristic: 
    how many distinct docs contributed to the score?
    More diversity = Low confidence in any single source

    Returns a score between 0 and 1
    """

    if not docs:
        return 0.0

    unique_sources = len(set(d.metadata.get("filename") for d in docs))
    avg_chunk_size = sum(len(d.page_content) for d in docs) / len(docs)

    # More unique sources = less focused = lower confidence
    source_penalty = 1.0 - (unique_sources - 1) * 0.15
    # Larger chunks = more context = higher confidence
    size_bonus = min(avg_chunk_size / 800, 1.0)

    confidence = (source_penalty * 0.7) + (size_bonus * 0.3)

    return round(min(max(confidence, 0.0), 1.0), 2)

def query_policy(
        question: str,
        vectorstore: Chroma,
        llm: ChatOpenAI,
        category_filter: str=None,
        issuer_filter: str=None,
        k: int = 5
    ) -> dict:
    """
    Query policy documents with optional metadata filtering.
        
    Filters allow enterprise users to scope queries:
    - category_filter: 'AI Governance', 'AI Policy', 'AI Ethics'
    - issuer_filter: 'NIST', 'White House', 'UNESCO'
     """
    # Build filter if provided
    where_filter = {}
    if category_filter:
        where_filter["category"] = category_filter
    if issuer_filter:
        where_filter["issuer"] = issuer_filter

    # Retrieve with or without filter
    if where_filter:
        docs = vectorstore.similarity_search(
            question,
            k=k,
            filter=where_filter
        )
    else:
        docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return {
            "answer": "No relevant policy documents found for this query.",
            "sources": [],
            "confidence": 0.0
        }

    context = format_policy_context(docs)
    confidence = compute_confidence(docs, question)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "confidence": confidence,
        "sources": [
            {
                "title": doc.metadata.get("title"),
                "issuer": doc.metadata.get("issuer"),
                "year": doc.metadata.get("year"),
                "page": doc.metadata.get("page"),
                "category": doc.metadata.get("category"),
                "excerpt": doc.page_content[:200]
            }
            for doc in docs
        ]
    }