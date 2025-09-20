# Simple RAG Test

A simple Retrieval-Augmented Generation (RAG) demo application that demonstrates how to build a question-answering system using LangChain, FAISS, and Hugging Face models.

## 🚀 Features

- **Two Approaches**: One-pager for easy collaboration + modular for production use
- **Self-Contained**: `app.py` works independently without other files
- **Modular Architecture**: Clean, well-structured code with separate modules
- **Easy Configuration**: Centralized configuration management
- **Multiple Interfaces**: Both CLI and programmatic interfaces
- **Error Handling**: Robust error handling and logging
- **Interactive Mode**: Real-time question answering
- **Demo Mode**: Predefined queries to showcase capabilities

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- 8GB+ RAM (for the large model)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd simple-rag-test
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python cli.py --help
   ```

## 🎯 Quick Start

### One-Pager Version (Easiest for collaboration)
```bash
cd one-pager
python app.py
```

### Modular Version - Demo Mode
```bash
cd modular
python cli.py --mode demo
```

### Modular Version - Interactive Mode
```bash
cd modular
python cli.py --mode interactive
```

## 📖 Usage

### Command Line Interface

The CLI provides several options:

```bash
python cli.py [OPTIONS]

Options:
  --mode {demo,interactive}    Run mode (default: demo)
  --model TEXT                 Language model to use (default: google/flan-t5-large)
  --embed-model TEXT          Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)
  --top-k INTEGER             Number of documents to retrieve (default: 3)
  --help                      Show help message
```

### Programmatic Usage

```python
# For modular version
from modular.rag_system import RAGSystem
from modular.config import DEMO_DOCUMENTS

# Initialize the RAG system
rag = RAGSystem()
rag.initialize(DEMO_DOCUMENTS)

# Ask a question
answer = rag.ask("When is Dr. Sarah available?")
print(answer)
```

## 🏗️ Architecture

The project offers two approaches:

### One-Pager Approach (`app.py`)
- **Single file**: Everything in one place for easy sharing and collaboration
- **Self-contained**: No external dependencies on other project files
- **Quick demo**: Perfect for Google Colab, Jupyter notebooks, or quick sharing
- **Easy to understand**: All code visible at once

### Project Structure
```
simple-rag-test/
├── one-pager/          # One-pager version
│   ├── app.py         # Self-contained RAG demo
│   └── README.md      # One-pager documentation
├── modular/           # Modular version
│   ├── cli.py         # Command-line interface
│   ├── rag_system.py  # Core RAG system
│   ├── config.py      # Configuration settings
│   └── README.md      # Modular documentation
├── requirements.txt   # Python dependencies
└── README.md         # Main documentation
```

### Key Components

- **`one-pager/app.py`**: Self-contained RAG demo - perfect for collaboration and quick demos
- **`modular/rag_system.py`**: Core RAG system class with proper error handling
- **`modular/config.py`**: Centralized configuration for models, parameters, and demo data
- **`modular/cli.py`**: Command-line interface with demo and interactive modes

## 🔧 Configuration

### Model Configuration

You can modify the models used in `config.py`:

```python
# Language model (for text generation)
MODEL_NAME = "google/flan-t5-large"  # or "google/flan-t5-base" for faster/smaller

# Embedding model (for document similarity)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Text Processing Configuration

```python
CHUNK_SIZE = 300        # Size of document chunks
CHUNK_OVERLAP = 30      # Overlap between chunks
DEFAULT_TOP_K = 3       # Number of documents to retrieve
MAX_NEW_TOKENS = 256    # Maximum tokens to generate
```

## 📊 Demo Data

The application comes with sample medical clinic data including:
- Doctor availability schedules
- Clinic hours and policies
- Insurance and appointment procedures
- Special events and services

## 🎮 Example Queries

Try these sample questions:

- "When is Dr. Sarah available?"
- "Can I see a doctor on Sunday?"
- "If I want a skin checkup on a Sunday, can I see a doctor?"
- "When should I bring insurance documents for a weekday appointment?"
- "If I want to see both a cardiologist and a dermatologist on the same day, which day should I visit?"

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error:**
   - Use a smaller model: `--model google/flan-t5-base`
   - Reduce `MAX_NEW_TOKENS` in config.py

2. **Slow Performance:**
   - Ensure you have a CUDA-compatible GPU
   - Use the base model instead of large: `--model google/flan-t5-base`

3. **Installation Issues:**
   - Make sure you have Python 3.8+
   - Try installing dependencies one by one
   - Check your internet connection for model downloads

### Performance Tips

- **GPU Usage**: The application automatically uses GPU if available
- **Model Selection**: Use `flan-t5-base` for faster inference, `flan-t5-large` for better quality
- **Batch Processing**: For multiple queries, initialize the system once and reuse it

## 🔄 Development

### Adding New Documents

To add your own documents:

```python
from rag_system import RAGSystem

# Your custom documents
custom_docs = [
    "Your document content here...",
    "Another document...",
]

# Initialize and add documents
rag = RAGSystem()
rag.initialize(custom_docs)

# Ask questions
answer = rag.ask("Your question here?")
```

### Extending the System

The modular design makes it easy to extend:

1. **New Retrieval Methods**: Modify the `retrieve_documents` method
2. **Different Models**: Change the model configuration
3. **Custom Preprocessing**: Override the text splitting logic
4. **Additional Interfaces**: Create new CLI commands or web interfaces

## 📚 Dependencies

- **LangChain**: Framework for building LLM applications
- **FAISS**: Efficient similarity search and clustering
- **Transformers**: Hugging Face transformers library
- **Sentence Transformers**: Embedding models
- **Accelerate**: Optimized model loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Hugging Face](https://huggingface.co/) for the pre-trained models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search

---

**Note**: This is a demo application. For production use, consider additional features like authentication, rate limiting, and more sophisticated error handling.

