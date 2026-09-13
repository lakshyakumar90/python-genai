from graph.query_analyzer import QueryAnalyzer


analyzer = QueryAnalyzer()


queries = [
    "What technologies am I using for my AI coding agent?",
    "Who am I working with?",
    "What do you know about my AI coding agent?",
    "What projects am I building?",
]


for query in queries:

    print("\n==============================")
    print("QUERY:")
    print(query)

    result = analyzer.analyze(query)

    print("\nANALYSIS:")
    print(result.model_dump_json(indent=2))