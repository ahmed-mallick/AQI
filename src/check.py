import requests, os
from dotenv import load_dotenv
load_dotenv() 

api_key = os.environ.get("HOPSWORKS_API_KEY")
print("Key loaded:", bool(api_key), "length:", len(api_key) if api_key else 0)

resp = requests.get(
    "https://app.hopsworks.ai/hopsworks-api/api/variables/hopsworks",
    headers={"Authorization": f"ApiKey {api_key}"},
    timeout=10
)
print(resp.status_code)
print(resp.text[:1000])