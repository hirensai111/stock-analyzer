# test_env.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("Testing environment variables:")
print(f"POLYGON_API_KEY: {'Found' if os.getenv('POLYGON_API_KEY') else 'NOT FOUND'}")
print(f"NEWSAPI_API_KEY: {'Found' if os.getenv('NEWSAPI_API_KEY') else 'NOT FOUND'}")
print(f"Current directory: {os.getcwd()}")

# Import api_config and check
from api_config import api_config
print("\nAPI Config Keys:")
for name, key in api_config.keys.items():
    print(f"  {name}: {'Found' if key else 'Not found'} ({len(key)} chars)")