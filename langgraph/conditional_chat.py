import json
from typing import Literal, Optional

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)


class State(TypedDict):
    user_input: str
    llm_output: Optional[str]
    is_good: Optional[bool]
    evaluation: Optional[str]


# --------------------------------------------------
# 1. Generate the answer
# --------------------------------------------------
def chatbot(state: State):
    response = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": state.get("user_input")}],
    )

    return {"llm_output": response.choices[0].message.content}


# --------------------------------------------------
# 2. AI evaluates the generated answer
# --------------------------------------------------
def evaluation_ai(state: State):
    user_question = state["user_input"]
    answer = state["llm_output"] or ""

    evaluation_prompt = f"""
    You are an AI answer evaluator.

    Evaluate the answer produced by another AI.

    USER QUESTION:
    {user_question}

    AI ANSWER:
    {answer}

    Determine whether the AI answer is good.

    Evaluate it based on:

    1. Correctness
    2. Relevance to the user's question
    3. Clarity
    4. Whether it actually answers the question
    5. Whether it contains obvious hallucinations or incorrect claims

    Return ONLY valid JSON in this exact format:

    {{
        "is_good": true,
        "evaluation": "Short explanation of why the answer is good or bad."
    }}

    "is_good" must be either true or false.
    """

    response = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt,
            }
        ],
    )

    raw_output = response.choices[0].message.content or ""

    try:
        result = json.loads(raw_output)

        return {
            "is_good": bool(result["is_good"]),
            "evaluation": result["evaluation"],
        }

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback if the local model doesn't return valid JSON
        return {
            "is_good": False,
            "evaluation": f"Evaluator returned an invalid response: {raw_output}",
        }


def endnode(state: State):
    return {
        "llm_output": state["llm_output"],
        "is_good": state["is_good"],
        "evaluation": state["evaluation"],
    }


# --------------------------------------------------
# Build graph
# --------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("evaluation_ai", evaluation_ai)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge(
    "chatbot",
    "evaluation_ai",
)

graph_builder.add_edge(
    "evaluation_ai",
    "endnode",
)

graph_builder.add_edge(
    "endnode",
    END,
)

graph = graph_builder.compile()


# --------------------------------------------------
# Run
# --------------------------------------------------

output = graph.invoke(
    {
        "user_input": "Explain quantum computing in one sentence.",
        "llm_output": None,
        "is_good": None,
        "evaluation": None,
    }
)

print("\n==============================")
print("USER QUESTION")
print("==============================")
print(output["user_input"])

print("\n==============================")
print("AI ANSWER")
print("==============================")
print(output["llm_output"])

print("\n==============================")
print("IS GOOD?")
print("==============================")
print(output["is_good"])

print("\n==============================")
print("AI EVALUATION")
print("==============================")
print(output["evaluation"])
