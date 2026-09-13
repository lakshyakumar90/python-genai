from memory.neo4j_graph import Neo4jGraph


class GraphRetriever:

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def retrieve(
        self,
        user_id: str,
        entities: list[str],
        intent: str = "general",
        max_hops: int = 2,
    ):

        results = []

        for entity in entities:

            traversal = self.graph.traverse(
                user_id=user_id,
                entity_name=entity,
                max_hops=max_hops,
            )

            for item in traversal:

                if self._matches_intent(item, intent):
                    results.append(item)

        return results

    def _matches_intent(
        self,
        item: dict,
        intent: str,
    ) -> bool:

        relation = item.get("relation", "")

        target_type = item.get(
            "target_type",
            "",
        )

        source_type = item.get(
            "start_entity_type",
            "",
        )

        if intent == "find_technologies":

            return (
                target_type == "Technology"
                or relation == "USES"
            )

        if intent == "find_people":

            return (
                target_type == "Person"
                or relation in {
                    "WORKS_WITH",
                    "KNOWS",
                    "COLLABORATES_WITH",
                }
            )

        if intent == "find_projects":

            return (
                target_type == "Project"
                or relation in {
                    "BUILDING",
                    "CREATING",
                    "WORKING_ON",
                }
            )

        if intent == "find_relationships":

            return True

        if intent == "search_memory":

            return True

        return True