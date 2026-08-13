from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph.runner import run_travel_graph
from app.services.chat_store import chat_store


class TravelPreferences(BaseModel):
    destination: Optional[str] = None
    origin: Optional[str] = None
    budget: Optional[str] = None
    currency: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    persons: Optional[str] = "1"


class HotelOption(BaseModel):
    name: str
    price_per_night: Optional[float] = None
    nights: Optional[int] = None
    total_stay_price: Optional[float] = None
    rating: Optional[float] = None
    destination: Optional[str] = None
    address: Optional[str] = None
    source: Optional[str] = None
    price_level: Optional[str] = None
    price_range: Optional[str] = None
    currency: Optional[str] = None
    price_source: Optional[str] = None
    price_basis: Optional[str] = None
    web_details: List[dict[str, str]] = []


class FlightOption(BaseModel):
    airline: str
    price: Optional[float] = None
    departure: Optional[str] = None
    arrival: Optional[str] = None
    return_departure: Optional[str] = None
    return_arrival: Optional[str] = None
    return_airline: Optional[str] = None
    destination: Optional[str] = None
    origin: Optional[str] = None
    source: Optional[str] = None
    currency: Optional[str] = None
    price_source: Optional[str] = None
    web_details: List[dict[str, str]] = []


class RestaurantOption(BaseModel):
    name: str
    cuisine: Optional[str] = None
    rating: Optional[float] = None
    address: Optional[str] = None
    price_level: Optional[str] = None
    price_range: Optional[str] = None
    estimated_price: Optional[float] = None
    currency: Optional[str] = None
    price_source: Optional[str] = None
    web_details: List[dict[str, str]] = []


class TravelResponse(BaseModel):
    user_id: str
    preferences: TravelPreferences
    itinerary: str
    flights: List[FlightOption] = []
    hotels: List[HotelOption] = []
    restaurants: List[RestaurantOption] = []
    dining_summary: Optional[str] = None
    route: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str
    preferences: Optional[TravelPreferences] = None


def parse_travel_prompt(message: str, default_prefs: Optional[TravelPreferences] = None) -> TravelPreferences:
    origin = default_prefs.origin if default_prefs and default_prefs.origin else None
    destination = default_prefs.destination if default_prefs and default_prefs.destination else None
    budget = default_prefs.budget if default_prefs and default_prefs.budget else None
    currency = default_prefs.currency if default_prefs and default_prefs.currency else None
    start_date = default_prefs.start_date if default_prefs and default_prefs.start_date else None
    end_date = default_prefs.end_date if default_prefs and default_prefs.end_date else None
    persons = default_prefs.persons if default_prefs and default_prefs.persons else "1"

    msg = message.strip()

    # Extract origin (e.g., "from London", "from New York")
    origin_match = re.search(r'\bfrom\s+([A-Za-z\s]+?)(?=\s+to|\s+starting|\s+for|\s+with|\s+budget|\,|\.|$)', msg, re.IGNORECASE)
    if origin_match and origin_match.group(1).strip().lower() not in ["the", "a", "my"]:
        origin = origin_match.group(1).strip()

    # Extract destination (e.g., "to Paris", "to Tokyo")
    dest_match = re.search(r'\bto\s+([A-Za-z\s]+?)(?=\s+from|\s+starting|\s+for|\s+with|\s+budget|\,|\.|$)', msg, re.IGNORECASE)
    if dest_match and dest_match.group(1).strip().lower() not in ["the", "a", "my"]:
        destination = dest_match.group(1).strip()

    # Currency extraction
    curr_match = re.search(r'(EUR|USD|GBP|INR|₹|€|\$|£)', msg, re.IGNORECASE)
    if curr_match:
        raw_c = curr_match.group(1).upper()
        symbol_map = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
        currency = symbol_map.get(raw_c, raw_c)

    # Persons / Travelers extraction (e.g., "for 2 persons", "3 people", "1 traveler")
    persons_match = re.search(r'\b(?:for\s+)?(\d+)\s*(?:person|persons|people|traveler|travelers|pax|adult|adults)\b', msg, re.IGNORECASE)
    if persons_match:
        persons = persons_match.group(1)

    # Budget extraction (e.g., "30k", "30K", "30000")
    k_budget_match = re.search(r'\b(\d+)\s*k\b', msg, re.IGNORECASE)
    if k_budget_match:
        budget = str(int(k_budget_match.group(1)) * 1000)
    else:
        digit_matches = re.findall(r'\b\d{3,6}\b', msg)
        if digit_matches:
            non_year_digits = [d for d in digit_matches if not (d.startswith("202") or d.startswith("203"))]
            if non_year_digits:
                budget = non_year_digits[0]

    # Date extraction (YYYY-MM-DD or 18 aug to 25 aug)
    MONTHS = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', msg)
    if len(dates) >= 2:
        start_date = dates[0]
        end_date = dates[1]
    elif len(dates) == 1:
        start_date = dates[0]
    else:
        month_matches = re.findall(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)(?:\s+(\d{4}))?', msg, re.IGNORECASE)
        parsed_dates = []
        current_year = datetime.now().year
        for day_str, month_str, year_str in month_matches:
            m_lower = month_str.lower()
            if m_lower in MONTHS:
                m_num = MONTHS[m_lower]
                d_num = int(day_str)
                y_num = int(year_str) if year_str else current_year
                parsed_dates.append(f"{y_num:04d}-{m_num:02d}-{d_num:02d}")
        if len(parsed_dates) >= 2:
            start_date = parsed_dates[0]
            end_date = parsed_dates[1]
        elif len(parsed_dates) == 1:
            start_date = parsed_dates[0]

    return TravelPreferences(
        origin=origin,
        destination=destination,
        budget=str(budget) if budget else None,
        currency=currency,
        start_date=start_date,
        end_date=end_date,
        persons=persons,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="AI Travel Planner API")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def health() -> HTMLResponse:
        html_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.post("/plan", response_model=TravelResponse)
    def plan_trip(payload: TravelPreferences, user_id: str = "demo-user") -> TravelResponse:
        result = run_travel_graph(
            user_id=user_id,
            preferences=payload.model_dump(),
        )
        return TravelResponse(
            user_id=user_id,
            preferences=payload,
            itinerary=result.get("itinerary", "No itinerary available."),
            flights=result.get("flights", []),
            hotels=result.get("hotels", []),
            restaurants=result.get("restaurants", []),
            dining_summary=result.get("dining_summary"),
            route=result.get("route"),
        )

    @app.post("/chat", response_model=TravelResponse)
    def chat_trip(payload: ChatMessageRequest, user_id: str = "demo-user") -> TravelResponse:
        # Load previous conversation history and preferences from InMemoryStore
        existing_history = chat_store.get_history(user_id)
        existing_prefs_dict = chat_store.get_preferences(user_id)
        existing_prefs_obj = TravelPreferences(**existing_prefs_dict) if existing_prefs_dict else payload.preferences

        parsed_prefs = parse_travel_prompt(payload.message, existing_prefs_obj)
        pref_dict = {k: v for k, v in parsed_prefs.model_dump().items() if v is not None}
        pref_dict["user_message"] = payload.message

        # Append new user message to session message list
        updated_messages = list(existing_history)
        updated_messages.append({"role": "user", "content": payload.message})

        result = run_travel_graph(
            user_id=user_id,
            preferences=pref_dict,
            messages=updated_messages,
        )

        assistant_itinerary = result.get("itinerary", "No response available.")
        updated_messages.append({"role": "assistant", "content": assistant_itinerary})

        # Persist updated session history and preferences in LangChain's InMemoryStore
        chat_store.save_session(user_id, updated_messages, pref_dict)

        return TravelResponse(
            user_id=user_id,
            preferences=parsed_prefs,
            itinerary=assistant_itinerary,
            flights=result.get("flights", []),
            hotels=result.get("hotels", []),
            restaurants=result.get("restaurants", []),
            dining_summary=result.get("dining_summary"),
            route=result.get("route"),
        )

    @app.post("/clear")
    def clear_session(user_id: str = "demo-user") -> dict:
        chat_store.clear(user_id)
        return {"status": "cleared", "user_id": user_id}

    return app


app = create_app()
