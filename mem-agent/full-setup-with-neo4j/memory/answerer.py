from openai import OpenAI


class MemoryAnswerer:

    def __init__(self):

        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    def answer(
        self,
        query: str,
        context: str,
    ) -> str:

        system_prompt = f"""
You are a helpful personal AI assistant.

You have access to memories about the user.

Use the memories when they are relevant.

IMPORTANT RULES:

1. Always answer the user's current message.
2. Never refuse to answer simply because there are no memories.
3. If the user is telling you something new, acknowledge it naturally.
4. If memories contain relevant information, use them.
5. Never invent personal facts.
6. If something is not known, say that you don't know.
7. Do not mention Qdrant, Neo4j, Mem0, embeddings,
   retrieval, or internal memory implementation unless
   the user asks about it.

KNOWN MEMORY:

{context}
"""

        response = self.client.chat.completions.create(
            model="gemma4:e2b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        return content.strip()