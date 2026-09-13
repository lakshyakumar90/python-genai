from graph.extractor import GraphExtractor


extractor = GraphExtractor()

result = extractor.extract(
    "I'm building an AI coding agent with Alice using Python and LangGraph."
)

print(result.model_dump_json(indent=2))