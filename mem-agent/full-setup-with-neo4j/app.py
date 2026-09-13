from memory.hybrid_memory import HybridMemory
from memory.context_builder import ContextBuilder
from memory.answerer import MemoryAnswerer


USER_ID = "lakshya"


memory = HybridMemory()
context_builder = ContextBuilder()
answerer = MemoryAnswerer()


try:

    while True:

        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {
            "exit",
            "quit",
        }:
            break

        # ========================================
        # 1. SEARCH MEMORY
        # ========================================

        result = memory.retrieve(
            user_id=USER_ID,
            query=user_input,
        )

        # ========================================
        # 2. FORMAT MEMORY
        # ========================================

        context = context_builder.build(
            semantic_results=result["semantic"],
            graph_results=result["graph"],
        )

        print()
        print("================================")
        print("MEMORIES")
        print("================================")
        print(context)

        # ========================================
        # 3. SEND MEMORY + MESSAGE TO AI
        # ========================================

        print()
        print("Generating response...")

        ai_response = answerer.answer(
            query=user_input,
            context=context,
        )

        print()
        print("Assistant:", ai_response)

        # ========================================
        # 4. SAVE MEMORY
        # ========================================

        print()
        print("================================")
        print("STORING MEMORY")
        print("================================")

        memory.remember(
            user_id=USER_ID,
            user_message=user_input,
            assistant_message=ai_response,
        )

        print()
        print("Memory stored successfully.")

finally:

    memory.close()