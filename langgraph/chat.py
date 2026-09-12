from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:e2b")

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}


def samplenode(state: State):
    print("\n\nInside samplenode node: ", state)
    return {"messages": ["Hi, this is a message from samplenode node"]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)

graph = graph_builder.compile()

updated_state = graph.invoke({"messages": ["Hello"]})
print("\n\nupdated_state: ", updated_state)
