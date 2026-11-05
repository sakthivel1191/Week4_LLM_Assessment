import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=API_KEY) if API_KEY else None
def ask_gemini(
    prompt,
    temperature=0,
    top_p=0.95,
    json_mode=False
):
    if client is None:
        return None, "API key not configured"
    try:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config
        )
        return response.text, None
    except Exception as error:
        return None, str(error)
def extract_json(text):
    if not text:
        raise ValueError("Empty response")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    return json.loads(text)
if __name__ == "__main__":
    answer, error = ask_gemini(
        "Explain artificial intelligence in two sentences."
    )
    if error:
        print("Gemini error:", error)
    else:
        print(answer)
        