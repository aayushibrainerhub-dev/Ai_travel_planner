import json
import os
import urllib.parse
import urllib.request
from datetime import date
from typing import Any, TypedDict

from app.tools.currency import convert_amount
from app.tools.web_search import extract_web_price, travel_web_details


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]
    hotels: list[dict]


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
                if "=" in stripped and stripped.split("=", 1)[0].strip() == "GROK_API_CLOUD_KEY":
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _format_price_range(price_range: dict | None) -> str | None:
    """Format the Google Places price range without inventing a nightly rate."""
    if not isinstance(price_range, dict):
        return None

    def format_money(value: dict | None) -> str | None:
        if not isinstance(value, dict):
            return None
        currency_code = value.get("currencyCode", "")
        units = value.get("units")
        nanos = value.get("nanos", 0)
        try:
            unit_amount = float(units)
            nano_amount = float(nanos)
        except (TypeError, ValueError):
            return None
        amount = unit_amount + (nano_amount / 1_000_000_000)
        return f"{currency_code} {amount:,.2f}".strip()

    start = format_money(price_range.get("startPrice"))
    end = format_money(price_range.get("endPrice"))
    if start and end:
        return f"{start} – {end}"
    return start or end


def _stay_nights(start_date: str, end_date: str) -> int | None:
    """Return check-out minus check-in in nights for ISO-formatted dates."""
    try:
        nights = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    except (TypeError, ValueError):
        return None
    return nights if nights > 0 else None


def _is_realistic_price(amount: float | None, currency: str) -> bool:
    """Check if per-night price amount is realistic for a real hotel stay."""
    if amount is None or amount <= 0:
        return False
    curr = (currency or "INR").upper().strip()
    if curr == "INR":
        return amount >= 1500.0
    if curr in ("USD", "EUR"):
        return amount >= 25.0
    if curr == "GBP":
        return amount >= 20.0
    return amount >= 10.0


def _generate_llm_hotel_options(
    *,
    destination: str,
    start_date: str,
    end_date: str,
    currency: str,
    user_query: str,
) -> list[dict[str, Any]]:
    """Ask the configured LLM for hotel estimates when the Places API has no data.

    These are planning estimates only; the LLM cannot confirm live inventory or prices.
    """
    api_key = _load_llm_api_key()
    if not api_key:
        return []

    search_query = (
        f"Hotels in {destination}, check-in {start_date}, check-out {end_date}, "
        f"priced in {currency}."
    )
    prompt = (
        f"Create planning-only hotel estimates for this travel request.\n"
        f"Destination: {destination}\n"
        f"Dates: {start_date} to {end_date}\n"
        f"Currency: {currency}\n\n"
        'Return ONLY valid JSON in this exact form:\n'
        '{"hotels":[{"name":"string","rating":4.5,"price_per_night":4500,"address":"string"}]}\n\n'
        f'Provide realistic per-night hotel rates in {currency} (e.g. 3500 to 20000 INR or 45 to 250 USD).\n'
        f"Return up to 4 real hotels in {destination}.\n"
    )
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
        print(f"LLM hotel estimate unavailable ({exc}).")
        return []

    options = generated.get("hotels", []) if isinstance(generated, dict) else []
    return [option for option in options[:4] if isinstance(option, dict)]


def search_hotels(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "destination")
    start_date = state.get("preferences", {}).get("start_date", "")
    end_date = state.get("preferences", {}).get("end_date", "")
    currency = state.get("preferences", {}).get("currency", "INR").upper()
    nights = _stay_nights(start_date, end_date)
    api_key = _load_api_key()
    user_query = state.get("preferences", {}).get("user_message", "")

    places = []
    api_error = None
    if api_key:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.priceRange",
            "Content-Type": "application/json"
        }
        text_query = (
            f"best hotels in {destination} "
            f"hotels resorts accommodation"
        )
        payload = json.dumps({
            "textQuery": text_query
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
            places = data.get("places", [])[:4]
        except Exception as e:
            api_error = str(e)
            print(f"Exception during Google Maps API call: {e}")
    else:
        api_error = "No Google Maps API key found."

    hotels = []
    if places:
        for place in places[:4]:
            name = place.get("displayName", {}).get("text", "Unnamed hotel")
            price_search_results = travel_web_details(
                f"{name} {destination} hotel price per night {start_date} to {end_date}",
                max_results=5,
            )
            web_price = extract_web_price(
                price_search_results,
                entity_name=name,
                default_currency=currency,
            )
            web_details = price_search_results[:1]
            price_per_night = None
            total_stay_price = None
            if web_price:
                src_curr = web_price.get("currency") or currency
                price_amt = web_price["amount"]
                if src_curr and src_curr.upper() != currency.upper():
                    price_amt = convert_amount(price_amt, src_curr, currency)

                price_kind = web_price.get("price_kind", "unspecified")
                if (price_kind == "total_stay" or (currency == "INR" and price_amt > 50000) or (currency in ("USD", "EUR", "GBP") and price_amt > 2500)) and nights and nights > 1:
                    calc_nightly = round(price_amt / nights, 2)
                    calc_total = round(price_amt, 2)
                else:
                    calc_nightly = round(price_amt, 2)
                    calc_total = round(calc_nightly * nights, 2) if nights else None

                if _is_realistic_price(calc_nightly, currency):
                    price_per_night = calc_nightly
                    total_stay_price = calc_total

            if price_per_night is None:
                price_per_night = 5500.0 if currency == "INR" else 75.0
                total_stay_price = round(price_per_night * nights, 2) if nights else None

            hotels.append(
                {
                    "name": name,
                    "price_per_night": price_per_night,
                    "total_stay_price": total_stay_price,
                    "currency": currency,
                    "rating": place.get("rating") or 4.5,
                    "destination": destination,
                    "address": place.get("formattedAddress", ""),
                }
            )

    # When the Places API is unavailable (quota exhausted, missing key, etc.)
    # or returns no results, ask the LLM to create request-specific planning
    # estimates so the user still receives hotel recommendations.
    if not hotels:
        print(f"Google Places API unavailable ({api_error or 'no places returned'}); using LLM planning estimates.")
        estimates = _generate_llm_hotel_options(
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            user_query=user_query,
        )[:4]
        for option in estimates[:4]:
            name = str(option.get("name", "")).strip()
            if not name:
                continue
            price_search_results = travel_web_details(
                f"{name} {destination} hotel price per night {start_date} to {end_date}",
                max_results=5,
            )
            web_price = extract_web_price(
                price_search_results,
                entity_name=name,
                default_currency=currency,
            )
            price_per_night = None
            total_stay_price = None

            if web_price:
                src_curr = web_price.get("currency") or currency
                price_amt = web_price["amount"]
                if src_curr and src_curr.upper() != currency.upper():
                    price_amt = convert_amount(price_amt, src_curr, currency)

                price_kind = web_price.get("price_kind", "unspecified")
                if (price_kind == "total_stay" or (currency == "INR" and price_amt > 50000) or (currency in ("USD", "EUR", "GBP") and price_amt > 2500)) and nights and nights > 1:
                    calc_nightly = round(price_amt / nights, 2)
                    calc_total = round(price_amt, 2)
                else:
                    calc_nightly = round(price_amt, 2)
                    calc_total = round(calc_nightly * nights, 2) if nights else None

                if _is_realistic_price(calc_nightly, currency):
                    price_per_night = calc_nightly
                    total_stay_price = calc_total

            if price_per_night is None:
                llm_val = option.get("price_per_night")
                if isinstance(llm_val, (int, float)) and _is_realistic_price(float(llm_val), currency):
                    price_per_night = float(llm_val)
                else:
                    price_per_night = 5500.0 if currency == "INR" else 75.0
                total_stay_price = round(price_per_night * nights, 2) if nights else None

            hotels.append(
                {
                    "name": name,
                    "price_per_night": price_per_night,
                    "total_stay_price": total_stay_price,
                    "currency": currency,
                    "rating": option.get("rating") or 4.5,
                    "destination": destination,
                    "address": str(option.get("address", "")),
                }
            )

    return {"hotels": hotels[:4]}