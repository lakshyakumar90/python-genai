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
    

def main():
    user_input = input("> ")
    response = client.chat.completions.create(
        model='gemma4:e2b',
        messages=[{"role": "user", "content": user_input}]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    # main()
    print(get_weather("goa"))

