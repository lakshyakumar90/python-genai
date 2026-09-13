import json

from openai import OpenAI
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    entities: list[str] = Field(default_factory=list)
    intent: str


class QueryAnalyzer:

    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    def analyze(self, query: str) -> QueryAnalysis:

        prompt = f"""
You are analyzing a query for a personal knowledge graph.

The graph belongs to:

User:
- Name: Lakshya
- User ID: lakshya

The graph may contain entities such as:

- People
- Projects
- Technologies
- Organizations
- Places
- Concepts

Your task is to determine:

1. Which EXISTING graph entities should be used as starting points.
2. What kind of information the user is asking for.

IMPORTANT:

The user may refer to themselves indirectly:

"I"
"me"
"my"
"mine"

These refer to Lakshya.

The user may also ask about categories rather than explicitly naming an entity.

Examples:

"What technologies am I using?"
→ entity: "Lakshya"
→ intent: "find_technologies"

"What projects am I building?"
→ entity: "Lakshya"
→ intent: "find_projects"

"Who am I working with?"
→ entity: "Lakshya"
→ intent: "find_people"

"What technologies am I using for my AI coding agent?"
→ entities:
   "Lakshya"
   "AI coding agent"
→ intent: "find_technologies"

"What do you know about my AI coding agent?"
→ entity:
   "AI coding agent"
→ intent: "search_memory"

"Who is Alice?"
→ entity:
   "Alice"
→ intent: "find_relationships"

IMPORTANT ENTITY RULES:

- Only return entities that are likely to exist in the graph.
- Do NOT return generic category words such as:
  "projects"
  "technologies"
  "people"
  "things"
  "memories"
- Resolve "I", "me", "my", and "mine" to "Lakshya".
- Do not invent entities.
- Keep entities canonical.
- Do not answer the question.

INTENTS:

search_memory
find_relationships
find_technologies
find_people
find_projects
general

Examples:

Query:
"What technologies am I using?"

Output:
{{
    "entities": ["Lakshya"],
    "intent": "find_technologies"
}}

Query:
"What projects am I building?"

Output:
{{
    "entities": ["Lakshya"],
    "intent": "find_projects"
}}

Query:
"Who am I working with?"

Output:
{{
    "entities": ["Lakshya"],
    "intent": "find_people"
}}

Query:
"What technologies am I using for my AI coding agent?"

Output:
{{
    "entities": ["Lakshya", "AI coding agent"],
    "intent": "find_technologies"
}}

Return ONLY valid JSON:

{{
    "entities": ["entity name"],
    "intent": "intent_name"
}}

User query:

{query}
"""

        response = self.client.chat.completions.create(
            model="gemma4:e2b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise knowledge graph query "
                        "analysis system. Never invent graph entities."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("LLM returned empty response.")

        data = json.loads(content)

        return QueryAnalysis.model_validate(data)