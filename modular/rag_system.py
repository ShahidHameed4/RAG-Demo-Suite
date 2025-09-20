"""
RAG (Retrieval-Augmented Generation) System implementation.
"""

import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from config import (
    MODEL_NAME, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, 
    DEFAULT_TOP_K, MAX_NEW_TOKENS
)


class RAGSystem:
    """
    A Retrieval-Augmented Generation system for question answering.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, embed_model: str = EMBED_MODEL):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Name of the language model to use
            embed_model: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embed_model = embed_model
        self.llm = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
    def load_llm(self) -> HuggingFacePipeline:
        """
        Load the language model.
        
        Returns:
            HuggingFacePipeline: The loaded language model
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, 
                device_map="auto"
            )
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            return self.llm
        except Exception as e:
            raise RuntimeError(f"Failed to load language model: {e}")
    
    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Build a FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            FAISS: The built vector store
        """
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            return self.vectorstore
        except Exception as e:
            raise RuntimeError(f"Failed to build vector store: {e}")
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document strings to add
        """
        try:
            # Convert strings to Document objects
            docs = [Document(page_content=doc) for doc in documents]
            
            # Split documents if needed
            split_docs = self.text_splitter.split_documents(docs)
            
            # Build vector store
            self.build_vectorstore(split_docs)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents: {e}")
    
    def retrieve_documents(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please add documents first.")
        
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            return retriever.get_relevant_documents(query)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve documents: {e}")
    
    def ask(self, query: str, top_k: int = DEFAULT_TOP_K, verbose: bool = True) -> str:
        """
        Ask a question and get an answer using RAG.
        
        Args:
            query: The question to ask
            top_k: Number of documents to retrieve
            verbose: Whether to print detailed output
            
        Returns:
            The answer to the question
        """
        if self.llm is None:
            raise ValueError("Language model not loaded. Please call load_llm() first.")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(query, top_k)
            
            # Step 2: Build context
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            # Step 3: Generate answer
            prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
            answer = self.llm(prompt)
            
            # Step 4: Print results if verbose
            if verbose:
                self._print_results(query, retrieved_docs, answer)
            
            return answer
            
        except Exception as e:
            raise RuntimeError(f"Failed to process query: {e}")
    
    def _print_results(self, query: str, retrieved_docs: List[Document], answer: str) -> None:
        """
        Print formatted results.
        
        Args:
            query: The original query
            retrieved_docs: Retrieved documents
            answer: The generated answer
        """
        print("=" * 80)
        print("Q:", query)
        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(f" [{i}] {doc.page_content}")
        print("\n--- Answer ---")
        print(answer)
        print("=" * 80)
    
    def initialize(self, documents: List[str]) -> None:
        """
        Initialize the RAG system with documents.
        
        Args:
            documents: List of document strings to index
        """
        print("Loading language model...")
        self.load_llm()
        
        print("Adding documents to vector store...")
        self.add_documents(documents)
        
        print("RAG system initialized successfully!")
