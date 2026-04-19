import sys
sys.path.append('.')

from src.ingestion.document_loader import load_single_PDF, load_directory, inspect_documents
from src.chunking.text_splitter import chunk_documents, inspect_chunks
from src.embedding.embedder import get_embedder, embed_text, inspect_embedding
from src.vectorstore.store import get_vectorstore, add_documents, similarity_search, inspect_results
from src.retrieval.rag_chain import get_llm, ask, print_response

# --- Build pipeline ---
docs = load_single_PDF("data/raw/Towards_an_ai_coscientist.pdf")
chunks = chunk_documents(docs)
embedder = get_embedder()
vectorstore = get_vectorstore(embedder)
add_documents(vectorstore, chunks)
llm = get_llm()

# --- Ask questions ---
questions = [
    "What evaluation metrics were used to assess the AI co-scientist?",
    "What are the main limitations of the system?",
    "What is the capital of France?"  # should say I don't know
]

for question in questions:
    print(f"\nQUESTION: {question}")
    response = ask(question, vectorstore, llm)
    print_response(response)

# docs = load_single_PDF("data/raw/Towards_an_ai_coscientist.pdf")
# # print(f"Pages loaded: {len(docs)}\n")

# chunks = chunk_documents(docs)
# print(f"Pages: {len(docs)} | Chunks: {len(chunks)}\n")

# embedder = get_embedder()

# # # We only test one chunk — embedding all 405 costs money
# # inspect_embedding(chunks[0].page_content, embedder)

# # # Test semantic similarity manually
# # print("\n--- Semantic similarity test ---")
# # v1 = embedder.embed_query("AI systems for scientific research")
# # v2 = embedder.embed_query("machine learning in science")
# # v3 = embedder.embed_query("football match results")

# # # Dot product as a rough similarity measure
# # def dot(a, b):
# #     return sum(x*y for x,y in zip(a,b))

# # print(f"Similar pair score: {dot(v1, v2):.4f}")
# # print(f"Dissimilar pair score: {dot(v1, v3):.4f}")

# vectorstore = get_vectorstore(embedder)

# add_documents(vectorstore, chunks)

# # Test retrieval
# query = "What evaluation metrics were used to assess the AI co-scientist?"
# results = similarity_search(vectorstore, query)
# inspect_results(results)