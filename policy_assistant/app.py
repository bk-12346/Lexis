import sys
sys.path.append('.')

import streamlit as st
from src.embedding.embedder import get_embedder
from src.retrieval.rag_chain import get_llm
from policy_assistant.core.ingestor import get_policy_vectorstore, ingest_policy_documents
from policy_assistant.core.retriever import query_policy
from policy_assistant.core.audit import log_query, get_recent_queries

# --- Page config ---
st.set_page_config(
    page_title="Policy Document Assistant",
    page_icon="📋",
    layout="wide"
)

# --- Initialize pipeline (cached so it only runs once) ---
@st.cache_resource
def initialize():
    embedder = get_embedder()
    vectorstore = ingest_policy_documents("data/policy_docs", embedder)
    llm = get_llm()
    return vectorstore, llm


# --- Sidebar ---
st.sidebar.title("🔍 Filter Options")
st.sidebar.markdown("Scope your query to specific documents")

category_filter = st.sidebar.selectbox(
    "Filter by Category",
    options=["All", "AI Governance", "AI Policy", "AI Ethics"]
)

issuer_filter = st.sidebar.selectbox(
    "Filter by Issuer",
    options=["All", "NIST", "White House", "UNESCO"]
)

username = st.sidebar.text_input("Your name (for audit log)", value="analyst")

st.sidebar.markdown("---")
st.sidebar.markdown("**Loaded Documents:**")
st.sidebar.markdown("- NIST AI Risk Management Framework")
st.sidebar.markdown("- US Executive Order on AI (2023)")
st.sidebar.markdown("- UNESCO AI Ethics Recommendation")

# --- Main UI ---
st.title("📋 Policy Document Assistant")
st.markdown("Query government AI policy documents with source attribution and audit logging.")

# Initialize
with st.spinner("Initializing pipeline..."):
    vectorstore, llm = initialize()

st.success("Pipeline ready. Ask a question below.")

# --- Query input ---
question = st.text_input(
    "Ask a policy question:",
    placeholder="e.g. What are the requirements for AI risk management?"
)

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    show_audit = st.button("Show Audit Log", use_container_width=True)

# --- Handle query ---
if submit and question:
    filters = {}
    if category_filter != "All":
        filters["category"] = category_filter
    if issuer_filter != "All":
        filters["issuer"] = issuer_filter

    with st.spinner("Searching policy documents..."):
        response = query_policy(
            question=question,
            vectorstore=vectorstore,
            llm=llm,
            category_filter=filters.get("category"),
            issuer_filter=filters.get("issuer")
        )

    # Log to audit
    log_query(
        user=username,
        question=question,
        answer=response["answer"],
        sources=response["sources"],
        confidence=response["confidence"],
        filters=filters
    )

    # --- Display answer ---
    st.markdown("### Answer")

    # Confidence indicator
    confidence = response["confidence"]
    if confidence >= 0.7:
        st.success(f"Confidence: {confidence:.0%}")
    elif confidence >= 0.4:
        st.warning(f"Confidence: {confidence:.0%} — answer may be incomplete")
    else:
        st.error(f"Confidence: {confidence:.0%} — limited relevant content found")

    st.markdown(response["answer"])

    # --- Sources ---
    st.markdown("### Sources")
    for i, source in enumerate(response["sources"]):
        with st.expander(f"[{i+1}] {source['title']} | Page {source['page']}"):
            st.markdown(f"**Issuer:** {source['issuer']}")
            st.markdown(f"**Year:** {source['year']}")
            st.markdown(f"**Category:** {source['category']}")
            st.markdown(f"**Excerpt:** {source['excerpt']}...")

# --- Audit log ---
if show_audit:
    st.markdown("### Recent Audit Log")
    entries = get_recent_queries(10)
    if not entries:
        st.info("No queries logged yet.")
    else:
        for entry in reversed(entries):
            with st.expander(f"{entry['timestamp']} | {entry['user']} | {entry['question'][:60]}..."):
                st.markdown(f"**Confidence:** {entry['confidence']}")
                st.markdown(f"**Filters:** {entry['filters_applied']}")
                st.markdown(f"**Sources:** {', '.join(entry['sources_cited'])}")
                st.markdown(f"**Answer:** {entry['answer'][:300]}...")