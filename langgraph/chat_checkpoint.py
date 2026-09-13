from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.checkpoint.mongodb import MongoDBSaver

llm = ChatOllama(model="gemma4:e2b")

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}



graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

DB_URI = "mongodb://localhost:27017/"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer)

    config = {"configurable": {"thread_id": "lakshya"}}
    
    for chunk in graph_with_checkpointer.stream({
        "messages": ["what is my name"]
    }, config=config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
