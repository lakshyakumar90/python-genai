from openai import OpenAI
import json

client = OpenAI(
    api_key="AIzaSyCm6vV9Ei9u5ilN4EeCC15UulonFnhEc3Q", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are an expert AI assistant in resolving user queries using chain of thought. You work on start, plan, and output steps. Full stop. You need to first plan what needs to be done. The plan can be multiple steps. Once you think enough plan has been done, comma, finally you can give an output. 

Output Format Rules:
- You MUST respond with EXACTLY ONE JSON object per turn (do not return a JSON array/list).
- The JSON object must strictly follow this structure:
  {"step": "PLAN" | "OUTPUT", "content": "String"}

Workflow Rules:
- Run one step per turn.
- Create multiple "PLAN" steps to break down the task.
- Once finished planning, produce a final "OUTPUT" step with the complete response for the user.


Example: 
    START: Hey can you solve 2 + 3 * 5 / 10
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in math problem"}
    PLAN: {"step": "PLAN", "content": "looking at the problem we should solve it using the BODMAS rule"}
    PLAN: {"step": "PLAN", "content": "Yes, the BODMAS rule is correct thing to be used here"}
    PLAN: {"step": "PLAN", "content": "First we must multiply 3 * 5 that is 15"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 15 / 10 "}
    PLAN: {"step": "PLAN", "content": "Then we must divide 15 / 10 that is 1.5"}
    PLAN: {"step": "PLAN", "content": "Now the new equation is 2 + 1.5"}
    PLAN: {"step": "PLAN", "content": "Now we must add 2 + 1.5 that is 3.5"}
    PLAN: {"step": "PLAN", "content": "Finally, the result is 3.5"}
    OUTPUT: {"step": "OUTPUT", "content": "The result is 3.5"}
"""

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_input = input("Enter your question: ")
message_history.append({"role": "user", "content": user_input})

while True:
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        response_format={"type": "json_object"},
        messages=message_history,
    )

    raw_result = response.choices[0].message.content
    message_history.append({"role": "assistant", "content": raw_result})
    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "START":
        print(f'STARTING: {parsed_result.get("content")}')
        continue

    if parsed_result.get("step") == "PLAN":
        print(f'THINKING: {parsed_result.get("content")}')
        continue

    if parsed_result.get("step") == "OUTPUT":
        print(f'OUTPUT: {parsed_result.get("content")}')
        break
    
