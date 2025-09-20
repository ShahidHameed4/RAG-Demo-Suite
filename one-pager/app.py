# ============================================
# RAG DEMO PIPELINE (Google Colab / GitHub Example)
# ============================================

# Install deps if not already done
# !pip install langchain langchain-community faiss-cpu sentence-transformers transformers accelerate

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "google/flan-t5-large"   # swap with flan-t5-base for faster/smaller
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# LLM SETUP
# -----------------------------
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


llm = load_llm(MODEL_NAME)


# -----------------------------
# EMBEDDINGS + VECTOR STORE
# -----------------------------
def build_vectorstore(docs, embed_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.from_documents(docs, embeddings)


# -----------------------------
# DOCUMENTS (demo clinic data)
# -----------------------------
raw_docs = [
    Document(page_content="Dr. Sarah Johnson is a cardiologist available on Mondays, Wednesdays, and Fridays."),
    Document(page_content="Dr. Emily Chen is a dermatologist available on Saturdays."),
    Document(page_content="Dr. Amit Verma is a pediatrician available on Tuesdays and Thursdays."),
    Document(page_content="The clinic is closed on Sundays and public holidays."),
    Document(page_content="Patients must check in at the reception desk 15 minutes before their appointment."),
    Document(page_content="The clinic partners with several insurance providers. Claims are processed only on weekdays."),
    Document(page_content="Vaccination drives are held on the first Saturday of every month."),
]

# Split long docs if needed
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = splitter.split_documents(raw_docs)

# Build FAISS vector store
vectorstore = build_vectorstore(docs, EMBED_MODEL)


# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
def hybrid_retrieve(query, top_k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.get_relevant_documents(query)


# -----------------------------
# ASK FUNCTION
# -----------------------------
def ask(query: str, top_k: int = 3):
    # Step 1: Retrieve docs
    retrieved = hybrid_retrieve(query, top_k=top_k)

    # Step 2: Build context
    context = "\n".join([d.page_content for d in retrieved])

    # Step 3: Ask LLM
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
    answer = llm(prompt)

    # Step 4: Pretty print
    print("="*80)
    print("Q:", query)
    print("\n--- Retrieved Context ---")
    for i, c in enumerate(retrieved, start=1):
        print(f" [{i}] {c.page_content}")
    print("\n--- Answer ---")
    print(answer)
    print("="*80)


# -----------------------------
# DEMO QUERIES
# -----------------------------
ask("When is Dr. Sarah available?")
ask("Can I see a doctor on Sunday?")
ask("If I want a skin checkup on a Sunday, can I see a doctor?")
ask("When should I bring insurance documents for a weekday appointment?")
ask("If I want to see both a cardiologist and a dermatologist on the same day, which day should I visit?")
