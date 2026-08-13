import os
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


class RouteState(TypedDict, total=False):
    route: str
    maps_url: str


def build_route(state: TravelState) -> RouteState:
    destination = state.get("preferences", {}).get("destination", "the city")
    api_key = _load_api_key()
    if api_key:
        maps_url = f"https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}"
    else:
        maps_url = "https://www.google.com/maps"

    return {
        "route": f"From the airport, take the express train to central {destination}, then walk to the hotel and nearby attractions.",
        "maps_url": maps_url,
    }
