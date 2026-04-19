import sys
sys.path.append('.')

from src.ingestion.document_loader import load_directory
from src.chunking.text_splitter import chunk_documents
from src.embedding.embedder import get_embedder
from src.vectorstore.store import get_vectorstore, add_documents
from src.retrieval.rag_chain import get_llm, format_context

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

SYSTEM_PROMPT = """You are a helpful document assistant. You answer questions strictly 
based on the provided context from the documents. 

RULES:
- Only use information from the provided context
- If the context doesn't contain the answer, say so clearly
- Always reference which page your answer comes from
- Keep answers concise and precise
- You have access to conversation history — use it to understand follow-up questions"""


def build_pipeline():
    """Initialize and return all pipeline components."""
    print("Initializing RAG pipeline...")
    
    docs = load_directory("data/raw", file_type="pdf")
    chunks = chunk_documents(docs)
    embedder = get_embedder()
    vectorstore = get_vectorstore(embedder)
    add_documents(vectorstore, chunks)
    llm = get_llm()
    
    print(f"Pipeline ready. Loaded {len(docs)} pages as {len(chunks)} chunks.")
    print("Type 'quit' to exit, 'clear' to reset conversation history.\n")
    
    return vectorstore, llm


def chat(question: str, history: list, vectorstore, llm) -> str:
    """
    Single turn of conversation.
    History is passed in so the LLM understands follow-up questions.
    """
    # Retrieve relevant chunks for this question
    retrieved_docs = vectorstore.similarity_search(question, k=5)
    context = format_context(retrieved_docs)
    
    # Build message history for the LLM
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add conversation history
    for turn in history:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    
    # Add current question with context
    current_message = f"""CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}"""
    
    messages.append(HumanMessage(content=current_message))
    
    response = llm.invoke(messages)
    return response.content


def main():
    vectorstore, llm = build_pipeline()
    history = []
    
    while True:
        try:
            question = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
            
        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            history = []
            print("Conversation history cleared.\n")
            continue
        
        print("\nAssistant: ", end="", flush=True)
        answer = chat(question, history, vectorstore, llm)
        print(answer)
        
        # Store turn in history
        history.append({"question": question, "answer": answer})
        
        # Keep history to last 5 turns to control context size
        if len(history) > 5:
            history.pop(0)
        
        print()


if __name__ == "__main__":
    main()