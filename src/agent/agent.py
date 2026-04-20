from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from src.agent.tools import compare_concepts, search_documents, summarize_topic
from src.embedding.embedder import get_embedder
from src.vectorstore.store import get_vectorstore
from dotenv import load_dotenv

from test_ingestion import vectorstore

load_dotenv()

# Module-level instances — injected into tools
_vectorstore = None
_llm = None

def get_vectorstore_instance() -> Chroma:
    """Return the initialized vectorstore instance"""
    if _vectorstore is None:
        raise RuntimeError("Agent not initialized. Call initialize_agent() first.")
    return _vectorstore

def get_llm_instance() -> ChatOpenAI:
    """Return the initialized LLM instance"""
    if _llm is None:
        raise RuntimeError("Agent not initialized. Call initialize_agent() first.")
    return _llm

def initialize_agent():
    """
    Initialize all the agent components and return the agent executor
    This is called once at startup
    """
    global _vectorstore, _llm

    print("Initializing agent...")

    embedder = get_embedder()
    _vectorstore = get_vectorstore(embedder)
    _llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

    tools = [search_documents, summarize_topic, compare_concepts]

    # create_react_agent from LangGraph handles the ReAct loop for us
    agent = create_react_agent(
        model = _llm,
        tools = tools
    )

    return agent

def run_agent(agent, question: str) -> str:
    """
    Run the agent on a single question.
    Returns the final answer as a string.
    """
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    
    # The last message is always the final answer
    return result["messages"][-1].content