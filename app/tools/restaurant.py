import json
import os
import urllib.parse
import urllib.request
from typing import Any, TypedDict

from app.tools.web_search import extract_web_price, travel_web_details


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


def _generate_llm_restaurant_options(
    *,
    destination: str,
    currency: str = "INR",
) -> list[dict[str, Any]]:
    """Ask the configured LLM for dining options when Places API is unavailable."""
    api_key = _load_llm_api_key()
    if not api_key:
        return []

    prompt = (
        f"Create planning-only dining options for this destination.\n"
        f"Destination: {destination}\n"
        f"Currency: {currency}\n\n"
        'Return ONLY valid JSON in this exact form:\n'
        '{"restaurants":[{"name":"string","cuisine":"string","rating":4.5,"budget_per_meal":"string","address":"string"}]}\n\n'
        f'Include an estimated budget_per_meal in {currency} for each option (e.g. "₹500 - ₹1,200 per meal" or "$15 - $30 per meal").\n'
        f"Return 4 popular, authentic real restaurants or cafes in {destination}.\n"
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
    except Exception as exc:
        print(f"LLM restaurant estimate unavailable ({exc}).")
        return []

    options = generated.get("restaurants", []) if isinstance(generated, dict) else []
    return [option for option in options[:4] if isinstance(option, dict)]


def _format_price_range(price_range: dict | None) -> str | None:
    """Format the price range returned by Google Places without estimating prices."""
    if not isinstance(price_range, dict):
        return None

    def format_money(value: dict | None) -> str | None:
        if not isinstance(value, dict):
            return None
        try:
            amount = float(value.get("units")) + float(value.get("nanos", 0)) / 1_000_000_000
        except (TypeError, ValueError):
            return None
        return f"{value.get('currencyCode', '')} {amount:,.2f}".strip()

    start = format_money(price_range.get("startPrice"))
    end = format_money(price_range.get("endPrice"))
    if start and end:
        return f"{start} – {end}"
    return start or end


def find_restaurants(state: TravelState) -> TravelState:
    destination = state.get("preferences", {}).get("destination", "the area")
    currency = state.get("preferences", {}).get("currency", "INR").upper()
    api_key = _load_api_key()

    restaurants = []
    seen_names = set()
    errors = []

    if api_key:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.primaryTypeDisplayName,places.priceLevel,places.priceRange",
            "Content-Type": "application/json"
        }

        # Search separately so cafés are not omitted by a restaurant-only query.
        for place_type in ("restaurants", "cafes"):
            payload = json.dumps({"textQuery": f"{place_type} in {destination}"}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                errors.append(str(exc))
                continue

            for place in data.get("places", [])[:2]:
                name = place.get("displayName", {}).get("text", "").strip()
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                primary_type = place.get("primaryTypeDisplayName", {}).get("text", place_type[:-1].title())
                web_details = travel_web_details(f"{name} {destination} restaurant menu price")
                web_price = extract_web_price(web_details)
                price_rng = _format_price_range(place.get("priceRange"))
                budget_per_meal = (
                    f"{currency} {web_price['amount']} per meal"
                    if web_price
                    else price_rng
                    if price_rng
                    else f"{'₹500 - ₹1,200' if currency == 'INR' else '$15 - $30'} per meal"
                )
                restaurants.append(
                    {
                        "name": name,
                        "cuisine": primary_type.replace("_", " ").title(),
                        "rating": place.get("rating"),
                        "address": place.get("formattedAddress", ""),
                        "price_level": place.get("priceLevel"),
                        "price_range": price_rng,
                        "budget_per_meal": budget_per_meal,
                        "estimated_price": web_price["amount"] if web_price else None,
                        "currency": web_price["currency"] if web_price else currency,
                        "price_source": "duckduckgo_snippet" if web_price else None,
                        "web_details": web_details,
                    }
                )

    if not restaurants:
        print(f"Google Places API unavailable ({errors[0] if errors else 'no results'}); using LLM dining estimates.")
        estimates = _generate_llm_restaurant_options(destination=destination, currency=currency)[:4]
        for option in estimates:
            name = str(option.get("name", "")).strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            cuisine = str(option.get("cuisine", "Local Dining")).title()
            budget_per_meal = str(
                option.get("budget_per_meal")
                or (f"{'₹500 - ₹1,200' if currency == 'INR' else '$15 - $30'} per meal")
            )
            restaurants.append(
                {
                    "name": name,
                    "cuisine": cuisine,
                    "rating": option.get("rating") or 4.5,
                    "address": str(option.get("address", f"{destination}")),
                    "price_level": None,
                    "price_range": None,
                    "budget_per_meal": budget_per_meal,
                    "estimated_price": None,
                    "currency": currency,
                    "price_source": "llm_planning_estimate",
                    "web_details": [],
                }
            )

    return {
        "restaurants": restaurants[:4],
        "dining_summary": f"Recommended dining spots and cafés in {destination}.",
    }
