from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.errors import GraphInterrupt

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
    import os, json as _json
    from pathlib import Path as _Path
    from langchain_openai import ChatOpenAI

    today = datetime.now()

    api_key = os.getenv("GROK_API_CLOUD_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        env_path = _Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and line.split("=", 1)[0].strip() == "GROK_API_CLOUD_KEY":
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    # Extract ONLY what's explicitly in the current message — no defaults seeded
    extracted: dict = {}
    if api_key:
        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
            max_tokens=150,
        )
        prompt = (
            f"Today is {today.strftime('%Y-%m-%d')}. "
            "Extract travel details from the message below. "
            "Reply with ONLY a JSON object with these keys: "
            "origin, destination, budget (number only, no symbol), "
            "currency (3-letter code e.g. INR/USD/EUR), persons (number), "
            "start_date (YYYY-MM-DD, the FIRST date mentioned), "
            "end_date (YYYY-MM-DD, the SECOND date mentioned). "
            "Rules: only extract what is explicitly stated in the message; "
            "preserve exact order of dates as they appear; "
            "if only one date is mentioned set only start_date or end_date based on context "
            "(e.g. 'return date' or 'end date' means end_date); "
            "for budget like '60k' return 60000; "
            "use null for any field not mentioned.\n\nMessage: "
            + message
        )
        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            text = response.content.strip()
            s, e = text.find("{"), text.rfind("}") + 1
            if s != -1 and e > s:
                extracted = _json.loads(text[s:e])
                print(f"[LLM extract] raw={extracted}")
        except Exception as ex:
            print(f"[LLM extract] failed: {ex}")

    # Python fallback: extract persons if LLM missed it
    import re
    if extracted.get("persons") is None:
        m = re.search(r'\b(\d+)\s*(?:person|people|travell?er|passenger|adult|pax)?s?\b'
                      r'|\b(?:person|people|travell?er|passenger|adult|pax)s?\s+(\d+)\b',
                      message, re.IGNORECASE)
        if m:
            extracted["persons"] = m.group(1) or m.group(2)

    # Apply defaults for fields NOT found in current message
    def _get(key: str, default):
        val = extracted.get(key)
        if val is not None and str(val).strip() not in ("", "null"):
            return str(val).strip()
        return default

    dp = default_prefs
    return TravelPreferences(
        origin      = _get("origin",      dp.origin      if dp else None),
        destination = _get("destination", dp.destination if dp else None),
        budget      = _get("budget",      dp.budget      if dp else None),
        currency    = _get("currency",    dp.currency    if dp else None),
        start_date  = _get("start_date",  dp.start_date  if dp else None),
        end_date    = _get("end_date",    dp.end_date    if dp else None),
        persons     = _get("persons",     dp.persons     if dp else "1") or "1",
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
        from datetime import date as _date
        today = _date.today()

        existing_history    = chat_store.get_history(user_id)
        existing_prefs_dict = chat_store.get_preferences(user_id)
        existing_prefs_obj  = TravelPreferences(**existing_prefs_dict) if existing_prefs_dict else payload.preferences

        updated_messages = list(existing_history)
        updated_messages.append({"role": "user", "content": payload.message})

        def _save_and_return(msg: str, prefs: dict) -> TravelResponse:
            updated_messages.append({"role": "assistant", "content": msg})
            chat_store.save_session(user_id, updated_messages, prefs)
            mp = TravelPreferences(**{k: v for k, v in prefs.items() if k in TravelPreferences.model_fields})
            return TravelResponse(user_id=user_id, preferences=mp, itinerary=msg, flights=[], hotels=[], restaurants=[])

        def _to_date(d: Optional[str]):
            if not d: return None
            try: return datetime.strptime(d.strip(), "%Y-%m-%d").date()
            except ValueError: return None

        # ── 1. Parse & merge preferences ────────────────────────────────
        parsed_prefs = parse_travel_prompt(payload.message, existing_prefs_obj)

        # Validate LLM-extracted dates before merging
        def _valid_iso(d: Optional[str]) -> bool:
            if not d: return True
            try:
                datetime.strptime(d.strip(), "%Y-%m-%d")
                return True
            except ValueError:
                return False

        bad_dates = []
        if not _valid_iso(parsed_prefs.start_date): bad_dates.append(f"**{parsed_prefs.start_date}** (departure)")
        if not _valid_iso(parsed_prefs.end_date):   bad_dates.append(f"**{parsed_prefs.end_date}** (return)")
        if bad_dates:
            pref_dict = {k: v for k, v in existing_prefs_dict.items() if v is not None}
            pref_dict["user_message"] = payload.message
            return _save_and_return(
                "⚠️ Invalid date(s): " + ", ".join(bad_dates) + ". Please enter real calendar dates.",
                pref_dict
            )

        pref_dict = {k: v for k, v in existing_prefs_dict.items() if v is not None}
        for k, v in parsed_prefs.model_dump().items():
            if v is None:
                continue
            if k == "persons":
                try:
                    if int(v) < 1:
                        continue
                except (ValueError, TypeError):
                    continue
            pref_dict[k] = v
        pref_dict["user_message"] = payload.message

        # Check date order after merge
        rs, re_ = _to_date(pref_dict.get("start_date")), _to_date(pref_dict.get("end_date"))
        if rs and re_ and re_ <= rs:
            pref_dict.pop("end_date", None)
            err = (f"⚠️ Return date **{re_.isoformat()}** must be after "
                   f"departure date **{rs.isoformat()}**. "
                   "Please enter a valid return date.")
            return _save_and_return(err, pref_dict)

        # ── 2. Past-date guard on merged prefs ───────────────────────────
        def _is_past(d: Optional[str]) -> bool:
            if not d: return False
            try: return datetime.strptime(d.strip(), "%Y-%m-%d").date() < today
            except ValueError: return False

        past_labels = []
        if _is_past(pref_dict.get("start_date")):
            past_labels.append(f"departure date **{pref_dict.pop('start_date')}**")
        if _is_past(pref_dict.get("end_date")):
            past_labels.append(f"return date **{pref_dict.pop('end_date')}**")
        if past_labels:
            label = " and ".join(past_labels)
            verb  = "is" if len(past_labels) == 1 else "are"
            err   = (f"⚠️ The {label} {verb} in the past (today is **{today.isoformat()}**). "
                     "Please enter future dates for your trip.")
            return _save_and_return(err, pref_dict)

        # ── 3. Completeness guard ──────────────────────────────────────
        required = {
            "origin":      "📍 Origin city",
            "destination": "🏙️ Destination city",
            "start_date":  "📅 Departure date (e.g. 18 Aug 2026)",
            "end_date":    "📅 Return date (e.g. 25 Aug 2026)",
            "budget":      "💰 Total budget (e.g. 60000 or 60k)",
            "persons":     "👥 Number of travellers",
        }
        missing = [label for field, label in required.items() if not pref_dict.get(field)]
        if missing:
            collected = ", ".join(f"{f}={pref_dict[f]}" for f in required if pref_dict.get(f))
            ask_msg = (
                "I need a few more details before I can build your itinerary:\n\n"
                + "\n".join(f"  • {m}" for m in missing)
                + (f"\n\n✅ Got so far: {collected}" if collected else "")
            )
            return _save_and_return(ask_msg, pref_dict)

        # ── 4. Run the graph ────────────────────────────────────────────
        result = {}
        try:
            result = run_travel_graph(user_id=user_id, preferences=pref_dict, messages=updated_messages)
            assistant_itinerary = result.get("itinerary") or ""
        except GraphInterrupt as exc:
            assistant_itinerary = ""
            try:
                interrupts = exc.args[0]
                if interrupts:
                    val = getattr(interrupts[0], "value", None)
                    if val:
                        assistant_itinerary = str(val)
            except Exception:
                pass
            assistant_itinerary = assistant_itinerary or str(exc)
            print(f"[GraphInterrupt] caught → {assistant_itinerary[:120]}")
        except Exception as exc:
            print(f"[Graph ERROR] {type(exc).__name__}: {exc}")
            raise

        if not assistant_itinerary:
            assistant_itinerary = "I'm sorry, something went wrong. Please try again."

        updated_messages.append({"role": "assistant", "content": assistant_itinerary})
        chat_store.save_session(user_id, updated_messages, pref_dict)
        merged_prefs = TravelPreferences(**{k: v for k, v in pref_dict.items() if k in TravelPreferences.model_fields})
        return TravelResponse(
            user_id=user_id,
            preferences=merged_prefs,
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
