import os
import sys
import json
import urllib.request
from app.tools.hotel import _load_llm_api_key

api_key = _load_llm_api_key()
if not api_key:
    print("No api key")
    sys.exit(1)

prompt = """Create planning-only hotel estimates for this travel request.
Original user request: 
Normalized hotel query: Hotels in Paris, check-in 2026-08-15, check-out 2026-08-20, priced in USD.

Return ONLY valid JSON in this exact form:
{"hotels":[{"name":"string","rating":number|null,"price_per_night":number|null,"address":"string"}]}

Return up to four plausible hotels. Do not claim availability or prices are live or confirmed.
Use null for any price you cannot estimate.
"""
payload = json.dumps({
    "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    "messages": [
        {"role": "system", "content": "You produce strictly valid JSON for a travel planner."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.2,
    "response_format": {"type": "json_object"},
}).encode("utf-8")
request = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=payload,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "AI-Travel-Planner/1.0",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        completion = json.loads(response.read().decode("utf-8"))
    content = completion["choices"][0]["message"]["content"]
    print("Content:", content)
    generated = json.loads(content)
    print("Generated:", generated)
except Exception as exc:
    print(f"Exception: {exc}")
