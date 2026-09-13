import json
import os

from dotenv import load_dotenv
from mem0 import MemoryClient
from openai import OpenAI

load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")

if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is not set in your .env file")

memory_client = MemoryClient(api_key=MEM0_API_KEY)

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

MODEL = "gemma4:e2b"
USER_ID = "lakshya"

while True:
    user_input = input("\nEnter your message: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    search_memory = memory_client.search(
        query=user_input,
        filters={"user_id": USER_ID},
    )

    results = search_memory.get("results", [])

    memories = [
        {
            "id": mem.get("id"),
            "memory": mem.get("memory"),
            "score": mem.get("score"),
        }
        for mem in results
    ]

    print("\nFound memories:")
    print(json.dumps(memories, indent=2))

    system_prompt = f"""
    You are a helpful AI assistant.

    Here is relevant memory about the user:

    {json.dumps(memories, indent=2)}

    Use these memories when they are relevant to the user's question.

    Do not mention that you are using a memory system.
    Do not invent facts that are not present in the user's message
    or the provided memories.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
    )

    ai_response = response.choices[0].message.content

    print("\nAssistant:")
    print(ai_response)

    memory_client.add(
        messages=[
            {
                "role": "user",
                "content": user_input,
            },
            {
                "role": "assistant",
                "content": ai_response,
            },
        ],
        user_id=USER_ID,
    )

    print("\nMemory added.")
