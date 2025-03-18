"""
Demonstrate the basic principles of RAG systems.
"""
import requests

def jaccard_similarity(query: str, document: str) -> float:
    """
    Calculate Jaccard similarity between a query and document.
    
    Args:
        query: The input query string
        document: The document to compare against
        
    Returns:
        float: Jaccard similarity score (0-1) representing text similarity
    """
    # normalise text
    query = set(query.lower().split(" "))
    document = set(document.lower().split(" "))

    # calculate intersection and union
    intersection = query.intersection(document)
    union = query.union(document)

    return len(intersection)/len(union)

def retrieve(query: str, external_resources: list) -> str:
    """
    Retrieve the most relevant document from external resources based on query similarity.
    
    Args:
        query: The user's question
        external_resources: List of document strings to search through
        
    Returns:
        str: The document with highest similarity to the query
    """
    # score docs against query
    similarities = [jaccard_similarity(query, doc) for doc in external_resources]
    # get the index of the maximum similarity
    max_index = similarities.index(max(similarities))
    # return the top matching document
    return external_resources[max_index]


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
    retrieved_context = retrieve(query, external_resources)

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

    SYSTEM_PROMPT = """
    You are a Python expert who replies concisely only on Python
    related questions in less than 100 words text.
    """
    USER_QUERY = "Who created Python (programming language)?"

    knowledge_base = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Python is known for its simplicity and readability.",
        "Python supports procedural, object-oriented, and functional programming.",
        "The Python Package Index (PyPI) is the official repository for 3rd-party Python software.",
        "Python uses indentation to define code blocks."
    ]

    answer = chat_complete(
        query=USER_QUERY,
        external_resources=knowledge_base,
        system_prompt=SYSTEM_PROMPT
    )

    print(answer)
