# Modular RAG System

This folder contains the production-ready, modular RAG system with proper architecture and error handling.

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run demo mode
python cli.py --mode demo

# Run interactive mode
python cli.py --mode interactive
```

## Contents

- **`cli.py`**: Command-line interface
- **`rag_system.py`**: Core RAG system implementation
- **`config.py`**: Configuration settings
- **`README.md`**: This file

## Features

- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Robust error handling and logging
- **CLI Interface**: Both demo and interactive modes
- **Configuration Management**: Easy to modify settings
- **Extensible**: Easy to add new features

## Perfect For

- Production applications
- Development and testing
- Custom implementations
- Team collaboration
- Long-term projects

## Usage

### Command Line Interface

```bash
# Demo mode (predefined queries)
python cli.py --mode demo

# Interactive mode (ask your own questions)
python cli.py --mode interactive

# Custom model
python cli.py --model google/flan-t5-base --mode demo
```

### Programmatic Usage

```python
from rag_system import RAGSystem
from config import DEMO_DOCUMENTS

# Initialize the RAG system
rag = RAGSystem()
rag.initialize(DEMO_DOCUMENTS)

# Ask a question
answer = rag.ask("Your question here?")
print(answer)
```

## Configuration

Modify `config.py` to change:
- Model settings
- Text processing parameters
- Demo data
- Retrieval settings
