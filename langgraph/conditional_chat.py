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
    retry_count: int


# --------------------------------------------------
# 1. Generate the answer
# --------------------------------------------------
def chatbot(state: State):
    user_input = state["user_input"]
    previous_answer = state.get("llm_output")
    evaluation = state.get("evaluation")
    retry_count = state.get("retry_count", 0)

    # --------------------------------------------------------
    # First generation
    # --------------------------------------------------------

    if retry_count == 0:
        prompt = f"""
        Answer the user's question accurately and clearly.

        USER QUESTION:
        {user_input}
        """
    else:
        prompt = f"""
        You previously generated an answer to the user's question,
        but another AI evaluator determined that the answer was not good.

        Your job is to regenerate a substantially improved answer.

        USER QUESTION:
        {user_input}

        PREVIOUS ANSWER:
        {previous_answer}

        EVALUATOR FEEDBACK:
        {evaluation}

        Instructions:

        1. Fix every problem identified by the evaluator.
        2. Do not repeat the same mistakes.
        3. Make sure the answer directly answers the user's question.
        4. Keep the answer clear and accurate.
        5. Return ONLY the new answer.
        """

    response = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "llm_output": response.choices[0].message.content,
        "retry_count": retry_count + 1,
    }


# --------------------------------------------------
# 2. AI evaluates the generated answer
# --------------------------------------------------
def evaluation_ai(state: State) -> Literal["endnode", "chatbot"]:
    user_input = state["user_input"]
    answer = state.get("llm_output") or ""
    retry_count = state.get("retry_count", 0)

    evaluation_prompt = f"""
    You are an expert AI evaluator.

    Your job is to determine whether the generated answer is good
    enough to be returned to the user.

    USER QUESTION:
    {user_input}

    GENERATED ANSWER:
    {answer}

    Evaluate the answer using these criteria:

    1. Correctness
    2. Relevance
    3. Completeness
    4. Clarity
    5. Whether it actually answers the user's question
    6. Whether it contains hallucinations or incorrect information

    If the answer is good enough, mark it as good.

    If the answer is not good enough, mark it as bad and explain
    exactly what needs to be improved.

    Return ONLY valid JSON:

    {{
        "is_good": true,
        "evaluation": "The answer is correct, relevant and sufficiently clear."
    }}

    OR:

    {{
        "is_good": false,
        "evaluation": "The answer does not explain X and incorrectly claims Y. It should..."
    }}
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
    print("\n================ EVALUATOR ================")
    print(raw_output)

    # --------------------------------------------------------
    # Parse evaluator response
    # --------------------------------------------------------

    try:
        result = json.loads(raw_output)

        is_good = bool(result["is_good"])
        evaluation = str(result["evaluation"])

    except (json.JSONDecodeError, KeyError, TypeError):
        # If evaluator itself fails, treat the answer as bad.
        is_good = False
        evaluation = (
            "Evaluator returned an invalid response. "
            "Regenerate the answer and try again."
        )

    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    state["is_good"] = is_good
    state["evaluation"] = evaluation

    # --------------------------------------------------------
    # Decide where the graph should go
    # --------------------------------------------------------

    if is_good:
        return "endnode"

    else:
        return "chatbot"


# ============================================================
# END NODE
# ============================================================


def endnode(state: State):

    return {
        "llm_output": state["llm_output"],
        "is_good": state["is_good"],
        "evaluation": state["evaluation"],
        "retry_count": state["retry_count"],
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("evaluator", evaluator)
graph_builder.add_node("endnode", endnode)


# START
graph_builder.add_edge(
    START,
    "chatbot",
)


# chatbot → evaluator
graph_builder.add_edge(
    "chatbot",
    "evaluator",
)


# evaluator → either END or regenerate
graph_builder.add_conditional_edges(
    "evaluator",
    lambda state: "endnode" if state["is_good"] else "chatbot",
    {
        "endnode": "endnode",
        "chatbot": "chatbot",
    },
)


# end
graph_builder.add_edge(
    "endnode",
    END,
)


graph = graph_builder.compile()


# ============================================================
# RUN
# ============================================================

output = graph.invoke(
    {
        "user_input": "Explain quantum computing in one sentence.",
        "llm_output": None,
        "is_good": None,
        "evaluation": None,
        "retry_count": 0,
    }
)


# ============================================================
# RESULT
# ============================================================

print("\n\n========================================")
print("FINAL RESULT")
print("========================================")

print("\nQuestion:")
print(output["user_input"])

print("\nFinal Answer:")
print(output["llm_output"])

print("\nIs Good:")
print(output["is_good"])

print("\nEvaluator Feedback:")
print(output["evaluation"])

print("\nRegeneration Attempts:")
print(output["retry_count"])