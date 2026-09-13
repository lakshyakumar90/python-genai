from graph.extractor import GraphExtractor
from memory.neo4j_graph import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

USER_ID = "lakshya"

text = (
    "I'm building an AI coding agent with Alice "
    "using Python and LangGraph."
)


extractor = GraphExtractor()
graph = Neo4jGraph()

try:
    # 1. Extract graph from user message
    extracted = extractor.extract(text)

    print("\n=== Extracted Graph ===")
    print(extracted.model_dump_json(indent=2))

    # 2. Persist extracted graph into Neo4j
    graph.upsert_graph(
        user_id=USER_ID,
        graph=extracted,
    )

    print("\n=== Graph persisted to Neo4j ===")

    # 3. Search the graph
    results = graph.search(
        user_id=USER_ID,
        query="Lakshya",
    )

    print("\n=== Entity Relationships ===")
    
    relationships = graph.get_entity_relationships(
        user_id=USER_ID,
        entity_name="AI coding agent",
    )
    
    for relationship in relationships:
        print(relationship)
    
    
    print("\n=== Graph Traversal ===")
    
    traversal = graph.traverse(
        user_id=USER_ID,
        entity_name="Lakshya",
        max_hops=2,
    )
    
    for item in traversal:
        print(item)

    print("\n=== Graph Search ===")

    for result in results:
        print(result)

finally:
    graph.close()