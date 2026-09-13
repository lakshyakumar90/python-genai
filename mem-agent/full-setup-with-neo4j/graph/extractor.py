import json

from openai import OpenAI

from .models import GraphExtraction


class GraphExtractor:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    def extract(self, text: str) -> GraphExtraction:

        prompt = f"""
        Extract a knowledge graph from the user's message.

        IMPORTANT:
        - Extract ONLY facts explicitly stated or directly expressed by the user.
        - Never invent facts.
        - Preserve the direction of relationships.
        - The source should be the entity performing/owning the action.
        - The target should be the entity receiving/being acted upon.
        - Resolve first-person references such as "I", "me", "my" to the supplied user identity.
        - Do not create relationships merely because two entities appear in the same sentence.

        The current user identity is:

        User ID: lakshya
        User name: Lakshya

        Example:

        User:
        "I am building an AI coding agent with Alice using Python and LangGraph."

        Correct interpretation:

        Lakshya --BUILDING--> AI coding agent
        Lakshya --WORKS_WITH--> Alice
        AI coding agent --USES--> Python
        AI coding agent --USES--> LangGraph

        Return JSON with exactly this structure:

        {{
            "entities": [
                {{
                    "name": "string",
                    "type": "Person | Organization | Project | Product | Place | Technology | Concept | Other"
                }}
            ],
            "relationships": [
                {{
                    "source": "entity name",
                    "relation": "RELATION_NAME",
                    "target": "entity name"
                }}
            ]
        }}

        Rules:

        - Relationship names must be uppercase.
        - Use underscores between words.
        - Do not create duplicate entities.
        - Every relationship source and target MUST exist in the entities list.
        - Do not infer relationships that aren't supported by the message.
        - Preserve relationship direction.
        - Use canonical entity names consistently.

        User message:

        {text}
        """

        response = self.client.chat.completions.create(
            model="gemma4:e2b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise knowledge graph extraction system. Extract only facts supported by the user's message. Never hallucinate entities or relationships.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        data = json.loads(content)

        return GraphExtraction.model_validate(data)
