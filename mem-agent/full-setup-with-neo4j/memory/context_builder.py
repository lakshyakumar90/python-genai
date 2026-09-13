class ContextBuilder:

    def build(
        self,
        semantic_results,
        graph_results,
    ) -> str:

        sections = []

        # ========================================
        # SEMANTIC MEMORY
        # ========================================

        semantic_lines = []

        for memory in semantic_results:

            memory_id = memory.get("id", "")
            memory_text = memory.get(
                "memory",
                "",
            )

            if not memory_text:
                continue

            semantic_lines.append(
                f"- [{memory_id}] {memory_text}"
            )

        if semantic_lines:

            sections.append(
                "SEMANTIC MEMORIES:\n"
                + "\n".join(semantic_lines)
            )

        # ========================================
        # GRAPH MEMORY
        # ========================================

        graph_lines = []

        for item in graph_results:

            source = item.get(
                "source"
            )

            relation = item.get(
                "relation"
            )

            target = item.get(
                "target"
            )

            if not source or not relation or not target:
                continue

            graph_lines.append(
                f"- {source} --{relation}--> {target}"
            )

        if graph_lines:

            sections.append(
                "GRAPH MEMORIES:\n"
                + "\n".join(graph_lines)
            )

        # ========================================
        # NO MEMORY
        # ========================================

        if not sections:

            return "No relevant memories were found."

        return "\n\n".join(sections)