import json
import os
import urllib.parse
import urllib.request
from typing import Any, TypedDict

from app.tools.web_search import extract_web_price, travel_web_details


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]


def _load_api_key() -> str | None:
    api_key = os.getenv("AVIATION_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    if "AVIATION_API_KEY" in line:
                        _, value = line.split("=", 1)
                        api_key = value.strip().strip('"').strip("'")
                        break
    return api_key

CITY_IATA_MAP = {
    "london": "LHR",
    "paris": "CDG",
    "new york": "JFK",
    "tokyo": "HND",
    "dubai": "DXB",
    "rome": "FCO",
    "berlin": "BER",
    "amsterdam": "AMS",
    "madrid": "MAD",
    "barcelona": "BCN",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "toronto": "YYZ",
    "sydney": "SYD",
    "singapore": "SIN",
    "hong kong": "HKG",
    "bangkok": "BKK",
    "istanbul": "IST",
    "seoul": "ICN",
    # Indian Cities
    "goa": "GOI",
    "ahmedabad": "AMD",
    "gujarat": "AMD",  # Fallback state to major city
    "mumbai": "BOM",
    "delhi": "DEL",
    "bangalore": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "pune": "PNQ",
    "kochi": "COK",
    "jaipur": "JAI",
}

def _get_iata(city_name: str) -> str:
    city_name_lower = city_name.strip().lower()
    
    # Exact match first
    mapped = CITY_IATA_MAP.get(city_name_lower)
    if mapped:
        return mapped
        
    # Substring match
    for key, iata in CITY_IATA_MAP.items():
        if key in city_name_lower:
            return iata
            
    # Fallback
    if len(city_name.strip()) == 3:
        return city_name.strip().upper()
    return city_name.strip().upper()[:3]

def _format_time(time_str: str) -> str:
    if not time_str:
        return "N/A"
    return time_str[:16].replace("T", " ")


def _load_llm_api_key() -> str:
    """Load the Groq/OpenAI-compatible key used by the travel agent."""
    api_key = os.getenv("GROK_API_CLOUD_KEY", "").strip().strip('"').strip("'")
    if api_key:
        return api_key

    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("GROK_API_CLOUD_KEY="):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _generate_llm_flight_options(
    *, origin: str, destination: str, start_date: str, end_date: str,
    currency: str, user_query: str,
) -> list[dict[str, Any]]:
    """Ask the configured LLM for itinerary estimates when the flight API has no data.

    These are planning estimates only; the LLM cannot confirm live inventory or prices.
    """
    api_key = _load_llm_api_key()
    if not api_key:
        return []

    search_query = (
        f"Round-trip flight options from {origin} to {destination}, departing "
        f"{start_date} and returning {end_date}, priced in {currency}."
    )
    prompt = f"""Create planning-only flight estimates for this travel request.
Original user request: {user_query or search_query}
Normalized flight query: {search_query}

Return ONLY valid JSON in this exact form:
{{"flights":[{{"airline":"string","departure":"YYYY-MM-DD HH:MM","arrival":"YYYY-MM-DD HH:MM","return_airline":"string","return_departure":"YYYY-MM-DD HH:MM","return_arrival":"YYYY-MM-DD HH:MM","price":number|null}}]}}

Return up to three plausible options. Do not claim the flights, flight numbers, times,
availability, or prices are live or confirmed. Use null for any price you cannot estimate.
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
        generated = json.loads(content)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"LLM flight estimate unavailable ({exc}).")
        return []

    options = generated.get("flights", []) if isinstance(generated, dict) else []
    return [option for option in options[:3] if isinstance(option, dict)]


def search_flights(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "destination")
    origin = state.get("preferences", {}).get("origin", "origin")
    api_key = _load_api_key()

    if not api_key:
        print("Aviation API key is not configured; using LLM planning estimates.")
        data = {}
    else:
        dep_iata = _get_iata(origin)
        arr_iata = _get_iata(destination)
        params = urllib.parse.urlencode({"access_key": api_key, "dep_iata": dep_iata, "arr_iata": arr_iata})
        url = f"https://api.aviationstack.com/v1/flights?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"Flight API unavailable/rate-limited ({e}). Using LLM planning estimates.")
            data = {}

    currency = state.get("preferences", {}).get("currency", "USD").upper()
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.0, "CAD": 1.36, "AUD": 1.52, "JPY": 155.0}
    rate = rates.get(currency, 1.0)
    start_date = state.get("preferences", {}).get("start_date", "")
    end_date = state.get("preferences", {}).get("end_date", "")
    date_str = start_date
    ret_date_str = end_date
    user_query = state.get("preferences", {}).get("user_message", "")
    flight_web_details = travel_web_details(
        f"round trip flight price from {origin} to {destination} {date_str} {ret_date_str}"
    )
    flight_web_price = extract_web_price(flight_web_details)

    flights = []
    if data and "data" in data and isinstance(data["data"], list):
        for flight in data.get("data", [])[:3]:
            airline_name = flight.get("airline", {}).get("name", "Airline")
            flight_iata = flight.get("flight", {}).get("iata", "")
            airline = f"{airline_name} ({flight_iata})" if flight_iata else airline_name
            
            dep_time = flight.get("departure", {}).get("scheduled", "")
            arr_time = flight.get("arrival", {}).get("scheduled", "")
            
            if start_date and "T" in dep_time:
                dep_time = f"{start_date}T{dep_time.split('T')[1]}"
            if start_date and "T" in arr_time:
                arr_time = f"{start_date}T{arr_time.split('T')[1]}"
                
            departure = _format_time(dep_time)
            arrival = _format_time(arr_time)
            
            raw_price = flight.get("price")
            price = int(raw_price * rate) if isinstance(raw_price, (int, float)) else None
            price_currency = currency
            price_source = "aviation_api" if price is not None else None
            if flight_web_price:
                price = flight_web_price["amount"]
                price_currency = flight_web_price["currency"]
                price_source = "duckduckgo_snippet"

            flights.append(
                {
                    "trip_type": "round_trip",
                    "airline": airline,
                    "price": price,
                    "currency": price_currency,
                    "price_source": price_source,
                    "departure": departure,
                    "arrival": arrival,
                    "return_airline": airline,
                    "return_departure": f"{ret_date_str} 17:30",
                    "return_arrival": f"{ret_date_str} 19:15",
                    "destination": destination,
                    "origin": origin,
                    "source": "aviation_api",
                    "web_details": flight_web_details,
                }
            )

    # When the live API fails, ask the LLM to create request-specific planning estimates.
    # Do not substitute hard-coded flight schedules, which are misleading for other routes.
    if not flights:
        estimates = _generate_llm_flight_options(
            origin=origin,
            destination=destination,
            start_date=date_str,
            end_date=ret_date_str,
            currency=currency,
            user_query=user_query,
        )
        for option in estimates:
            flights.append(
                {
                    "trip_type": "round_trip",
                    "airline": str(option.get("airline", "Estimated airline")),
                    "price": flight_web_price["amount"] if flight_web_price else (option.get("price") if isinstance(option.get("price"), (int, float)) else None),
                    "currency": flight_web_price["currency"] if flight_web_price else currency,
                    "price_source": "duckduckgo_snippet" if flight_web_price else "llm_planning_estimate",
                    "departure": str(option.get("departure", "N/A")),
                    "arrival": str(option.get("arrival", "N/A")),
                    "return_airline": str(option.get("return_airline", option.get("airline", "Estimated airline"))),
                    "return_departure": str(option.get("return_departure", "N/A")),
                    "return_arrival": str(option.get("return_arrival", "N/A")),
                    "destination": destination,
                    "origin": origin,
                    "source": "llm_planning_estimate",
                    "web_details": flight_web_details,
                }
            )

    return {"flights": flights}
