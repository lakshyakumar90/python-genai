from mem0 import Memory


class Mem0Memory:

    def __init__(self):

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
                    "ollama_base_url": "http://localhost:11434",
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

        self.memory = Memory.from_config(config)

    def search(
        self,
        user_id: str,
        query: str,
    ):

        return self.memory.search(
            query=query,
            filters={
                "user_id": user_id,
            },
        )

    def add(
        self,
        user_id: str,
        messages: list[dict],
    ):

        return self.memory.add(
            user_id=user_id,
            messages=messages,
        )