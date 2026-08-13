import json
import os
import urllib.parse
import urllib.request
from typing import TypedDict


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]


def _load_api_key() -> str | None:
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    if "OPEN_WEATHER_API_KEY" in line:
                        _, value = line.split("=", 1)
                        api_key = value.strip().strip('"').strip("'")
                        break
    return api_key


def get_weather(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "your destination")
    api_key = _load_api_key()

    if not api_key:
        return {"weather": "Weather unavailable because OPEN_WEATHER_API_KEY is missing."}

    query = urllib.parse.urlencode({"q": destination, "appid": api_key, "units": "metric"})
    url = f"https://api.openweathermap.org/data/2.5/weather?{query}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"weather": f"Unable to fetch weather for {destination}: {exc}"}

    if data.get("cod") not in (200, "200"):
        message = data.get("message", "unknown error")
        return {"weather": f"Weather lookup failed for {destination}: {message}"}

    main_data = data.get("main", {})
    weather_data = data.get("weather", [{}])[0]
    temperature = main_data.get("temp")
    description = weather_data.get("description", "clear skies").capitalize()
    humidity = main_data.get("humidity")

    weather_text = (
        f"{description} in {destination} with {temperature}°C"
        + (f", humidity {humidity}%" if humidity is not None else "")
    )
    return {"weather": weather_text, "weather_raw": data}
