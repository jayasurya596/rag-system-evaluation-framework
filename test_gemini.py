import os
from google import genai

def main():
    print("Checking if GEMINI_API_KEY is in environment:", "GEMINI_API_KEY" in os.environ)
    try:
        # Initialize the client. If no key is set, it will try to find it in the env
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Hello, say "Connection Successful!"',
        )
        print("Success! Gemini responded:")
        print(response.text)
    except Exception as e:
        print("Failed to initialize or call Gemini API:")
        print(e)

if __name__ == "__main__":
    main()
