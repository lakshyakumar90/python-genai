from graph.query_analyzer import QueryAnalyzer
from graph.retriever import GraphRetriever
from memory.neo4j_graph import Neo4jGraph


USER_ID = "lakshya"


query = "What technologies am I using for my AI coding agent?"


analyzer = QueryAnalyzer()
graph = Neo4jGraph()
retriever = GraphRetriever(graph)


try:

    print("\n=== USER QUERY ===")
    print(query)

    # 1. Analyze query
    analysis = analyzer.analyze(query)

    print("\n=== QUERY ANALYSIS ===")
    print(analysis.model_dump_json(indent=2))

    # 2. Retrieve graph information
    graph_results = retriever.retrieve(
        user_id=USER_ID,
        entities=analysis.entities,
        max_hops=2,
    )

    print("\n=== GRAPH RESULTS ===")

    for result in graph_results:
        print(result)

finally:
    graph.close()