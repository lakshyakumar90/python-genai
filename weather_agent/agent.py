import json
from openai import OpenAI
import requests

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama', 
)

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t+(%f)+%w+%h+%p"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Error: Unable to fetch weather data."

available_tools = {
    "get_weather": get_weather,
}

SYSTEM_PROMPT = """
You are an expert AI assistant in resolving user queries using chain of thought. You work on start, plan, tool, and output steps. You need to first plan what needs to be done. The plan can be multiple steps. Once you think enough plan has been done, finally you can give an output. 
You can also call a tool if required from the list of available tools.
For every tool call, wait for the observe step which is the output from the tool call.

Rules:
- Strictly follow the output JSON format.
- Run one step at a time.
- The sequence of steps is START (when user gives a query), PLAN (break down the task), TOOL (use a tool), and OUTPUT (give the final response).

Output JSON Format:
  {"step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool": "string", "input": "string"}

Available Tools:
  - get_weather(city: str) -> str: Fetches the weather for a given city.

Example 1:
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

Example 2:
    START: What is the weather of delhi
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in weather in delhi india"}
    PLAN: {"step": "PLAN", "content": "lets see if we have any available tools from the list of available tools"}
    PLAN: {"step": "PLAN", "content": "Great, we have get_weather tool available for this query"}
    PLAN: {"step": "PLAN", "content": "I need to call the get_weather tool for delhi as input for city"}
    PLAN: {"step": "TOOL", "tool": "get_weather", "content": "delhi"}
    PLAN: {"step": "OBSERVE", "content": "The weather in delhi is cloudy with a chance of rain and temperature of 30°C"}
    PLAN: {"step": "PLAN", "content": "I got the weather info of delhi"}
    OUTPUT: {"step": "OUTPUT", "content": "The weather in delhi is cloudy with a chance of rain and temperature of 30°C"}
"""

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

while True:
    user_input = input("Enter your question: ")
    message_history.append({"role": "user", "content": user_input})
    
    while True:
        response = client.chat.completions.create(
            model="gemma4:e2b",
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
    
        if parsed_result.get("step") == "TOOL":
            tool_to_call = parsed_result.get("tool")
            tool_input = parsed_result.get("input")
            print(f'CALLING TOOL: {tool_to_call} with input: {tool_input}')
            tool_response = available_tools[tool_to_call](tool_input)
            message_history.append({"role": "developer", "content": json.dumps({
                "step": "OBSERVE",
                "tool": tool_to_call,
                "input": tool_input,
                "output": tool_response,
            })})
            continue
    
        if parsed_result.get("step") == "OUTPUT":
            print(f'OUTPUT: {parsed_result.get("content")}')
            break
        
