import json
import os
import urllib.parse
import urllib.request
from typing import TypedDict


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]


def _load_api_key() -> str | None:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    if "GOOGLE_MAPS_API_KEY" in line:
                        _, value = line.split("=", 1)
                        api_key = value.strip().strip('"').strip("'")
                        break
    return api_key


def search_hotels(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "destination")
    api_key = _load_api_key()

    if not api_key:
        print("No API key found.")
        return {"hotels": []}

    print("API key found.", api_key)
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"textQuery": f"hotels in {destination}"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Exception during Google Maps API call: {e}")
        return {"hotels": []}
    
    currency = state.get("preferences", {}).get("currency", "USD").upper()
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.0, "CAD": 1.36, "AUD": 1.52, "JPY": 155.0}
    rate = rates.get(currency, 1.0)

    places = data.get("places", [])[:4]
    hotels = []
    for place in places:
        name = place.get("displayName", {}).get("text", "Unnamed hotel")
        base_usd = 120 + len(hotels) * 30
        converted_price = int(base_usd * rate)
        if currency == "INR":
            converted_price = round(converted_price / 100) * 100

        hotels.append(
            {
                "name": name,
                "price_per_night": converted_price,
                "currency": currency,
                "rating": place.get("rating", 4.0),
                "destination": destination,
                "address": place.get("formattedAddress", ""),
                "source": "google_maps",
            }
        )

    if not hotels:
        return {"hotels": []}

    return {"hotels": hotels}
