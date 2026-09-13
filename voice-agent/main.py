import os

import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from openai import OpenAI


load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

elevenlabs = ElevenLabs(
    api_key=os.environ.get("ELEVENLABS_API_KEY"),
)


SYSTEM_PROMPT = """
You are an expert voice agent.

The user is speaking to you through a voice interface.
Your response will be converted directly into speech.

Rules:
- Speak naturally and conversationally.
- Keep responses concise.
- Do not use markdown.
- Do not use bullet points.
- Do not use emojis.
- Output only what should be spoken aloud.
"""
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

def main():
    r = sr.Recognizer()
    
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2
            print("Listening...")
            audio = r.listen(source)
        print("Processing Audio (STT)...")
    
        try:
            stt = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return
        except sr.RequestError as e:
            print("Speech recognition error:", e)
            return
    
        print("STT Result:", stt)
    
        messages.append({"role": "user", "content": stt})
    
        print("Thinking...")
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
        )
        result = response.choices[0].message.content
        print("Response Content:", result)
    
        messages.append({"role": "assistant", "content": result})
    
        print("Generating speech...")
    
        audio = elevenlabs.text_to_speech.convert(
            text=result,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_v3",
            output_format="mp3_44100_128",
        )
        print("Speaking...")
        play(audio)


if __name__ == "__main__":
    main()