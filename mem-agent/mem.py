import json

from mem0 import Memory
from openai import OpenAI

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "gemma4:e2b",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
}

memory_client = Memory.from_config(config)

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

while True:
    user_input = input("Enter your message: ")

    search_memory = memory_client.search(
        query=user_input, filters={"user_id": "lakshya"}
    )

    memories = [
        f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}"
        for mem in search_memory.get("results", [])
    ]

    print("Found memories:", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )

    ai_response = response.choices[0].message.content
    print(ai_response)

    memory_client.add(
        user_id="lakshya",
        messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response},
        ],
    )

    print("Memory added")
