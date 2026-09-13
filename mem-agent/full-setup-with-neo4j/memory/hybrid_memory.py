from memory.mem0_memory import Mem0Memory
from memory.neo4j_graph import Neo4jGraph
from graph.extractor import GraphExtractor


class HybridMemory:

    def __init__(self):

        self.semantic = Mem0Memory()

        self.graph = Neo4jGraph()

        self.extractor = GraphExtractor()

    def close(self):

        self.graph.close()

    # ============================================
    # RETRIEVE
    # ============================================

    def retrieve(
        self,
        user_id: str,
        query: str,
    ):

        print("\nSearching semantic memory...")

        semantic_response = self.semantic.search(
            user_id=user_id,
            query=query,
        )

        semantic_results = semantic_response.get(
            "results",
            [],
        )

        print(
            f"Semantic memories found: "
            f"{len(semantic_results)}"
        )

        print("\nSearching graph memory...")

        graph_results = self.graph.get_user_graph(
            user_id=user_id,
        )

        print(
            f"Graph relationships found: "
            f"{len(graph_results)}"
        )

        return {
            "semantic": semantic_results,
            "graph": graph_results,
        }

    # ============================================
    # REMEMBER
    # ============================================

    def remember(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ):

        print("\nSaving semantic memory to Qdrant...")

        semantic_result = self.semantic.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                },
            ],
        )

        print("Semantic memory saved.")

        # ========================================
        # Extract graph
        # ========================================

        print("\nExtracting graph memory...")

        extracted = self.extractor.extract(
            user_message
        )

        print(
            "\nExtracted graph:"
        )

        print(
            extracted.model_dump_json(
                indent=2
            )
        )

        # ========================================
        # Save graph
        # ========================================

        print("\nSaving graph to Neo4j...")

        self.graph.upsert_graph(
            user_id=user_id,
            graph=extracted,
        )

        print("Graph memory saved.")

        return {
            "semantic": semantic_result,
            "graph": extracted,
        }