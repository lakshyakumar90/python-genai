from memory.neo4j_graph import Neo4jGraph
from graph.retriever import GraphRetriever


USER_ID = "lakshya"


graph = Neo4jGraph()
retriever = GraphRetriever(graph)


try:

    entities = [
        "AI coding agent"
    ]

    results = retriever.retrieve(
        user_id=USER_ID,
        entities=entities,
        max_hops=2,
    )

    print("\n=== GRAPH RETRIEVAL ===")

    for result in results:
        print(result)

finally:
    graph.close()