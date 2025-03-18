"""
Simple RAG-based question answering system using semantic retrieval and local LLM inference.
This program demonstrates a basic Retrieval-Augmented Generation (RAG) system that:

1. Takes a user query
2. Retrieves relevant context from a knowledge base using semantic similarity
3. Sends the query plus context to a local LLM server (Ollama)
4. Returns the generated response

The system uses cross-encoders for semantic search and the Ollama API for generation.
"""
import requests
from sentence_transformers.cross_encoder import CrossEncoder

def semantic_retrieval(
        query: str, corpus: list[str], score_threshold: float = 0.5
) -> list[str]:
    """
    Retrieve relevant documents from a corpus based on semantic similarity to the query.

    Args:
        query: The user question to find relevant documents for
        corpus: List of text documents to search through
        score_threshold: Minimum similarity score (0-1) for a document to be considered relevant
    
    Returns:
        str: Concatenated string of all relevant documents that meet the threshold
    """
    # 1. Load a pretrained CrossEncoder model
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    # 2. Rank all sentences in the corpus for the query
    ranks = model.rank(query, corpus)

    # 3. Retrieve the most relevant documents
    relevant_docs = [
        corpus[rank["corpus_id"]] for rank in ranks if rank["score"] >= score_threshold
    ]

    return ' '.join(relevant_docs)


def chat_complete(query: str, external_resources: list, system_prompt: str) -> str:
    """
    Generate a response to the query using RAG (Retrieval-Augmented Generation).
    
    Args:
        query: The user's question
        external_resources: List of document strings to retrieve context from
        system_prompt: Instructions for the LLM behavior
        
    Returns:
        str: The generated response from the language model
    """
    server_url = "http://localhost:11434/api/generate"
    retrieved_context = semantic_retrieval(query=query, corpus=external_resources)

    augemented_prompt = f"""Based on the following information:
    
    {retrieved_context}

    provide an answer to the following question:

    {query}
    """

    payload = {
    "model": "gemma3:4b",
    "prompt": augemented_prompt,
    "system": system_prompt,
    "stream": False
    }

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", server_url, headers=headers, json=payload, timeout=180)
    response = response.json()

    return response.get("response", None)

if __name__ == "__main__":

    knowledge_base = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Python is known for its simplicity and readability.",
        "Python supports procedural, object-oriented, and functional programming.",
        "The Python Package Index (PyPI) is the official repository for 3rd-party Python software.",
        "Python uses indentation to define code blocks.",
        "Python can be used for developing web applications.",
        "Python can be used for creating machine learning models."
    ]

    SYSTEM_PROMPT = """
    You are a Python expert who replies concisely only on Python
    related questions in less than 100 words text.

    You do not reply with code blocks.

    Your role is to answer theoretical and technical questions just verbally.
    """
    # USER_QUERY = "Who created Python (programming language)?"
    # USER_QUERY = "What are some use cases of Python?"

    USER_QUERY = input("Ask your Local Python Assistant:")

    answer = chat_complete(
        query=USER_QUERY,
        external_resources=knowledge_base,
        system_prompt=SYSTEM_PROMPT
    )

    print(answer)
