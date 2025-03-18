# Simple RAG Systems

A collection of Retrieval-Augmented Generation (RAG) implementations using Python and Ollama. This repository demonstrates how to build basic RAG systems.

## 📋 Overview

This repository contains implementations of two basic RAG systems:

1. **Simple RAG (00-simple-rag)**: Uses Jaccard similarity for basic keyword-based retrieval
2. **Semantic RAG (01-semantic-rag)**: Uses neural embeddings for semantic search and retrieval

All implementations connect to a locally running Ollama server for language model inference.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running locally
- A compatible LLM model loaded in Ollama (default: `gemma3:4b`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vasilogi/simple-rag.git
   cd simple-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama server (if not already running):
   ```bash
   ollama serve
   ```

5. Make sure you have the required model:
   ```bash
   ollama pull gemma3:4b
   ```

## 💻 Usage

### Running the Simple RAG System

```bash
cd 00-simple-rag
python main.py
```

### Running the Semantic RAG System

```bash
cd 01-semantic-rag
python main.py
```

For both implementations, you'll be prompted to enter your query, and the system will:
1. Find the most relevant information from the knowledge base
2. Send this information along with your query to the language model
3. Display the generated response

## 🔍 How It Works

### Simple RAG (Jaccard Similarity)

The basic implementation uses Jaccard similarity to match queries with relevant documents:
- Tokenizes both query and documents into word sets
- Calculates similarity based on shared words
- Selects the document with highest similarity score
- Augments the prompt with this document as context

### Semantic RAG (Neural Embeddings)

The advanced implementation uses neural embeddings for semantic search:
- Leverages a pre-trained cross-encoder model
- Captures semantic meaning beyond keyword matching
- Ranks documents by semantic relevance
- Filters results based on a similarity threshold

## ⚙️ Customization

You can customize the systems by:
- Modifying the knowledge base in the `knowledge_base` list
- Adjusting the system prompt in the `SYSTEM_PROMPT` variable
- Changing the LLM model in the `payload` dictionary
- Tuning the similarity threshold (for semantic RAG)

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Learn more about RAG systems](https://arxiv.org/abs/2005.11401)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

This README file has been generated with the help of Claude 3.7 Sonnet.
