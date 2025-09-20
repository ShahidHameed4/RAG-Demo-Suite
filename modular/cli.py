"""
Command-line interface for the RAG demo application.
"""

import argparse
import sys
from typing import List

from rag_system import RAGSystem
from config import DEMO_DOCUMENTS, DEMO_QUERIES


def run_demo_queries(rag_system: RAGSystem) -> None:
    """
    Run the predefined demo queries.
    
    Args:
        rag_system: Initialized RAG system
    """
    print("\n" + "="*80)
    print("RUNNING DEMO QUERIES")
    print("="*80)
    
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n[Demo Query {i}/{len(DEMO_QUERIES)}]")
        rag_system.ask(query)


def interactive_mode(rag_system: RAGSystem) -> None:
    """
    Run the RAG system in interactive mode.
    
    Args:
        rag_system: Initialized RAG system
    """
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\n> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
                
            rag_system.ask(query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="RAG Demo - A simple Retrieval-Augmented Generation system"
    )
    parser.add_argument(
        "--mode", 
        choices=["demo", "interactive"], 
        default="demo",
        help="Run mode: 'demo' for predefined queries, 'interactive' for user input"
    )
    parser.add_argument(
        "--model", 
        default="google/flan-t5-large",
        help="Language model to use (default: google/flan-t5-large)"
    )
    parser.add_argument(
        "--embed-model", 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=3,
        help="Number of documents to retrieve (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag_system = RAGSystem(model_name=args.model, embed_model=args.embed_model)
        rag_system.initialize(DEMO_DOCUMENTS)
        
        # Run based on mode
        if args.mode == "demo":
            run_demo_queries(rag_system)
        else:
            interactive_mode(rag_system)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
