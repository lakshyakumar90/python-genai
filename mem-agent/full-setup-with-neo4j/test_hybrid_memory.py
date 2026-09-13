from memory.hybrid_memory import HybridMemory


USER_ID = "lakshya"


memory = HybridMemory()


try:

    query = (
        "What technologies am I using "
        "for my AI coding agent?"
    )

    result = memory.retrieve(
        user_id=USER_ID,
        query=query,
    )

    print("\n==============================")
    print("QUERY")
    print("==============================")

    print(result["query"])

    print("\n==============================")
    print("QUERY ANALYSIS")
    print("==============================")

    print(
        result["analysis"].model_dump_json(
            indent=2
        )
    )

    print("\n==============================")
    print("SEMANTIC MEMORY")
    print("==============================")

    for item in result["semantic"]:
        print(item)

    print("\n==============================")
    print("GRAPH MEMORY")
    print("==============================")

    for item in result["graph"]:
        print(item)

finally:

    memory.close()