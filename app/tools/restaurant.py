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


def find_restaurants(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "the area")
    api_key = _load_api_key()

    if not api_key:
        return {
            "restaurants": [],
            "dining_summary": f"No restaurants found near {destination}.",
        }

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.primaryTypeDisplayName",
        "Content-Type": "application/json"
    }
    payload = json.dumps({"textQuery": f"restaurants in {destination}"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "restaurants": [],
            "dining_summary": f"Unable to fetch restaurants for {destination}: {exc}",
        }

    places = data.get("places", [])[:4]
    restaurants = []
    for place in places:
        name = place.get("displayName", {}).get("text", "Unnamed place")
        primary_type = place.get("primaryTypeDisplayName", {}).get("text", "Restaurant").replace("_", " ").title()
        restaurants.append(
            {
                "name": name,
                "cuisine": primary_type,
                "rating": place.get("rating", 4.0),
                "address": place.get("formattedAddress", ""),
            }
        )

    if not restaurants:
        return {
            "restaurants": [],
            "dining_summary": f"No restaurants found near {destination}.",
        }

    return {
        "restaurants": restaurants,
        "dining_summary": f"Recommended restaurants near {destination}.",
    }
