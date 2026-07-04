import os
from dotenv import load_dotenv
from xai_sdk import Client

load_dotenv()

print("Testing API Key...")
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY")
)

print("✅ Keys loaded successfully!")
print("API Key starts with:", os.getenv("XAI_API_KEY")[:10] + "...")
print("Management Key starts with:", os.getenv("XAI_MANAGEMENT_API_KEY")[:10] + "..." if os.getenv("XAI_MANAGEMENT_API_KEY") else "None")