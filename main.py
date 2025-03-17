import requests

# Create a simple knowledge base
# Implement basic retrival based on jaccard similarity
# Combine Retrieval and Generation steps

# user input + retrieved documents (context)
# Implement a basic retrieval based on jaccard similarity

def jaccard_similarity(query: str, document: str):
    # normalise text
    query = set(query.lower().split(" "))
    document = set(document.lower().split(" "))

    # calculate intersection and union
    intersection = query.intersection(document)
    union = query.union(document)

    return len(intersection)/len(union)

# Score similarities
def retrieve(query: str, knowledge_base: list):
    # score all docs against the query
    similarities = [jaccard_similarity(query, doc) for doc in knowledge_base]
    # get the index of the maximum similarity
    max_index = similarities.index(max(similarities))
    # return the top matching document
    return knowledge_base[max_index]

# Ollama server
def chat_complete(query: str, knowledge_base: list, system_prompt: str):
    # Interact with the server
    URL = "http://localhost:11434/api/generate"

    retrieved_context = retrieve(query, knowledge_base)

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

    response = requests.request("POST", URL, headers=headers, json=payload, timeout=180)
    result = response.json()

    return result.get("response", None)

if __name__ == "__main__":

    system_prompt = "You are a Python expert who replies concisely only on Python related questions in less than 100 words text."
    user_query = "Who created Python (programming language)?"

    knowledge_base = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Python is known for its simplicity and readability.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "The Python Package Index (PyPI) is the official repository for third-party Python software.",
        "Python uses indentation to define code blocks, unlike many other programming languages that use curly braces."
    ]

    response = chat_complete(query=user_query, knowledge_base=knowledge_base, system_prompt=system_prompt)

    print(response)