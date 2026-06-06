"""
LLM Connectivity Diagnostic
---------------------------
Verifies that the Google Gemini API key is configured correctly and the LLM
can be instantiated and invoked.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load local environment variables from .env
load_dotenv()

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GOOGLE_API_KEY environment variable is missing.")
        print("Please create a '.env' file in this directory and add your key:")
        print("GOOGLE_API_KEY=AIzaSy...")
        return

    print("GOOGLE_API_KEY found. Connecting to Gemini API...")
    try:
        # Instantiate model
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=api_key
        )
        
        # Test invocation
        response = llm.invoke('Say hello and confirm API connectivity in one sentence.')
        print("\nSuccess! Response from Gemini:")
        print(f"-> {response.content.strip()}")
    except Exception as e:
        print(f"\n[Error] Connection test failed: {e}")

if __name__ == "__main__":
    main()