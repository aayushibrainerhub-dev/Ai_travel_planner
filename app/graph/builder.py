"""
builder.py — LangGraph + LangChain Agent

Architecture:
    START → validate_dates → agent_node → END

Checkpointer: MemorySaver (required for interrupt() to work correctly).

validate_dates:
  Only fires when the user has provided dates AND they are in the past.
  Uses interrupt() to pause the graph and ask for corrected future dates.
  Greetings / partial messages pass straight through to agent_node.

agent_node:
  Runs the LangChain create_agent tool-calling loop. The agent’s system
  prompt already enforces collecting origin, destination, dates, budget,
  and person count before calling any travel tools.
"""

import json
import time
from datetime import date, datetime
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.cache.redis import RedisCache
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import CachePolicy, interrupt

from app.services.redis_client import get_client, set_key, get_key
from app.agent import build_travel_agent


# ---------------------------------------------------------------------------
# State & Context
# ---------------------------------------------------------------------------

class TravelState(TypedDict, total=False):
    preferences: dict[str, str]
    messages: list[dict]   # ← full conversation history [{role, content}, ...]
    # Populated after the agent runs
    origin: str
    destination: str
    start_date: str
    end_date: str
    flights: list[dict]
    hotels: list[dict]
    restaurants: list[dict]
    weather: str
    budget: str
    route: str
    dining_summary: str
    itinerary: str          # ← final answer from the agent


class TravelContext(TypedDict):
    user_id: str


# ---------------------------------------------------------------------------
# Redis cache (optional — graceful fallback when Redis is not available)
# ---------------------------------------------------------------------------

try:
    _redis_client = get_client()
    redis_cache = RedisCache(_redis_client)
    _redis_client.ping()
    print(f"[Redis] ✅ Connected successfully — host={_redis_client.connection_pool.connection_kwargs.get('host', '?')} port={_redis_client.connection_pool.connection_kwargs.get('port', '?')}")
    print("[Redis] ✅ LangGraph caching is ENABLED")

    set_key("redis_test", "working", ex=60)
    _val = get_key("redis_test")
    _ttl = _redis_client.ttl("redis_test")
    print(f"[Redis] ✅ set_key/get_key test — stored and retrieved: '{_val}' (TTL: {_ttl}s)")

    print("[Redis] 🔍 All keys currently stored in Redis:")
    try:
        _all_keys = _redis_client.keys("*")
        if _all_keys:
            for _k in _all_keys:
                _k_str   = _k.decode("utf-8", errors="replace") if isinstance(_k, bytes) else _k
                _dtype   = _redis_client.type(_k)
                _dtype   = _dtype.decode() if isinstance(_dtype, bytes) else _dtype
                _key_ttl = _redis_client.ttl(_k)
                if _dtype == "string":
                    _raw = _redis_client.get(_k)
                    if isinstance(_raw, bytes):
                        try:
                            _v_str = _raw.decode("utf-8")[:80]
                        except UnicodeDecodeError:
                            _v_str = f"<binary {len(_raw)} bytes: {_raw[:16].hex()}...>"
                    else:
                        _v_str = str(_raw)[:80]
                else:
                    _v_str = f"<{_dtype} type — not a string>"
                print(f"   [{_dtype.upper():6s}] {_k_str!r:45s}  TTL:{_key_ttl:5d}s  VALUE: {_v_str}")
        else:
            print("   (no keys found)")
    except Exception as _dump_err:
        print(f"   [Redis key dump error: {_dump_err}]")

except Exception as e:
    redis_cache = None
    print(f"[Redis] ❌ Connection failed — {type(e).__name__}: {e}")
    print("[Redis] ⚠️  Running WITHOUT cache (app still works normally)")


# ---------------------------------------------------------------------------
# Validation node 1: completeness check
# ---------------------------------------------------------------------------

def validate_completeness(state: TravelState) -> TravelState:
    """
    Interrupts if any required travel field is missing.
    Required: origin, destination, start_date, end_date, budget.
    'persons' defaults to '1' so it is never considered missing.
    """
    prefs = state.get("preferences", {})

    missing = []
    if not prefs.get("origin"):
        missing.append("📍 **Origin city** — where are you travelling *from*?")
    if not prefs.get("destination"):
        missing.append("🏙️ **Destination city** — where do you want to go?")
    if not prefs.get("start_date"):
        missing.append("📅 **Departure date** — when does your trip start? (e.g. *1 Aug 2027*)")
    if not prefs.get("end_date"):
        missing.append("📅 **Return date** — when does your trip end? (e.g. *8 Aug 2027*)")
    if not prefs.get("budget"):
        missing.append("💰 **Budget** — what is your total trip budget? (e.g. *60000 INR* or *60k*)")

    if missing:
        lines = "\n".join(f"  {item}" for item in missing)
        persons_hint = (
            "\n\n👥 *(Also let me know how many travellers if it’s more than 1.)*"
            if prefs.get("persons", "1") == "1" else ""
        )
        message = (
            "I still need a few details before I can build your itinerary:\n\n"
            + lines
            + persons_hint
            + "\n\nPlease share the missing info and I’ll get your full travel plan ready!"
        )
        interrupt(message)

    return state


# ---------------------------------------------------------------------------
# Validation node 2: date-range check
# ---------------------------------------------------------------------------

def _parse_date_safe(date_str: str | None) -> date | None:
    """Parse a 'YYYY-MM-DD' string into a date; return None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def validate_dates(state: TravelState) -> TravelState:
    """
    Validates that travel dates are not in the past.

    - Checks start_date and end_date from preferences.
    - If either date is before today, calls interrupt() which pauses
      the LangGraph execution and returns the error message to the
      /chat API endpoint so it can be shown to the user.
    - If dates are valid (or not provided), the state passes through
      unchanged and agent_node runs normally.
    """
    prefs     = state.get("preferences", {})
    today     = date.today()
    raw_start = prefs.get("start_date") or ""
    raw_end   = prefs.get("end_date")   or ""

    start_d = _parse_date_safe(raw_start)
    end_d   = _parse_date_safe(raw_end)

    past_labels = []
    if start_d and start_d < today:
        past_labels.append(f"departure date **{raw_start}**")
    if end_d and end_d < today:
        past_labels.append(f"return date **{raw_end}**")

    if past_labels:
        label_str = " and ".join(past_labels)
        message = (
            f"⚠️ The {label_str} {'is' if len(past_labels) == 1 else 'are'} in the past "
            f"(today is **{today.isoformat()}**). "
            "Please enter future dates for your trip."
        )
        # interrupt() pauses the graph here; the message is surfaced to
        # the caller by the /chat endpoint as the assistant reply.
        interrupt(message)

    # Dates are valid (or not provided) — continue to agent_node unchanged.
    return state


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def agent_node(state: TravelState, runtime: dict | None = None) -> TravelState:
    """
    Single LangGraph node that runs the LangChain tool-calling agent sub-graph.

    The agent (langchain 1.x create_agent):
      1. Receives user travel preferences as a natural-language message.
      2. Autonomously decides which tools to call (weather/flights/hotels…).
      3. Loops until no more tool calls are needed.
      4. Returns the final itinerary as the last message content.

    Tool outputs are extracted from the message history and stored back
    into TravelState as structured fields (flights, hotels, etc.).
    """
    prefs = state.get("preferences", {})
    history = state.get("messages", [])   # full conversation history from InMemoryStore
    raw_user_message = prefs.get("user_message", "")

    # Build a minimal message list for the agent:
    # System context (prefs) + only the latest user message.
    # Previous conversation history is NOT sent to avoid TPM overflow —
    # all required details are already in the system context message.
    known = {k: v for k, v in prefs.items() if v and k != "user_message"}
    context_line = "Trip details: " + ", ".join(f"{k}={v}" for k, v in known.items()) + "."

    # Get the last user message to send
    last_user_msg = raw_user_message
    if not last_user_msg and history:
        for m in reversed(history):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break
    if not last_user_msg:
        last_user_msg = (
            f"Plan a complete trip from {prefs.get('origin')} to {prefs.get('destination')}. "
            f"Dates: {prefs.get('start_date')} to {prefs.get('end_date')}. "
            f"Budget: {prefs.get('budget')} {prefs.get('currency','INR')}. "
            f"Persons: {prefs.get('persons','1')}."
        )

    trimmed = [
        {"role": "system", "content": context_line},
        {"role": "user",   "content": last_user_msg},
    ]

    import hashlib
    history_repr = json.dumps([(m.get("role"), m.get("content")) for m in trimmed])
    _cache_key = "travel_plan:" + hashlib.md5(history_repr.encode()).hexdigest()
    _CACHE_TTL = 3600

    if redis_cache is not None:
        _cached_raw = _redis_client.get(_cache_key)
        if _cached_raw:
            _cached = json.loads(_cached_raw.decode() if isinstance(_cached_raw, bytes) else _cached_raw)
            _ttl_left = _redis_client.ttl(_cache_key)

            print(f"\n{'='*60}")
            print(f"[Redis] ⚡ CACHE HIT — serving from Redis")
            print(f"[Redis] ⚡ KEY      : {_cache_key!r}")
            print(f"[Redis] ⚡ TTL LEFT : {_ttl_left}s ({_ttl_left // 60}m {_ttl_left % 60}s)")
            print(f"[Redis] ⚡ FIELDS   : {list(_cached.keys())}")
            print(f"{'─'*60}")

            _itin = _cached.get("itinerary", "")
            print(f"[Redis] 📋 ITINERARY ({len(_itin)} chars):")
            print(f"         {_itin[:200].replace(chr(10), ' | ')}...")

            _flights = _cached.get("flights", [])
            print(f"[Redis] ✈️  FLIGHTS  ({len(_flights)} found):")
            for i, f in enumerate(_flights):
                print(f"         [{i+1}] {f.get('airline','?')} | {f.get('departure','?')} → {f.get('arrival','?')} | {f.get('currency','?')} {f.get('price','?')}")

            _hotels = _cached.get("hotels", [])
            print(f"[Redis] 🏨 HOTELS   ({len(_hotels)} found):")
            for i, h in enumerate(_hotels):
                print(f"         [{i+1}] {h.get('name','?')} | ⭐{h.get('rating','?')} | {h.get('currency','?')} {h.get('price_per_night','?')}/night")
            _rests = _cached.get("restaurants", [])
            print(f"[Redis] 🍽️  RESTAURANTS ({len(_rests)} found):")
            for i, r in enumerate(_rests):
                print(f"         [{i+1}] {r.get('name','?')} | {r.get('cuisine','?')} | ⭐{r.get('rating','?')}")

            print(f"[Redis] 🌤️  WEATHER  : {str(_cached.get('weather',''))[:100]}")
            print(f"[Redis] 💰 BUDGET   : {str(_cached.get('budget',''))[:100]}")
            print(f"[Redis] 🗺️  ROUTE    : {str(_cached.get('route',''))[:100]}")
            print(f"{'='*60}\n")

            return _cached
        else:
            print(f"\n[Redis] 🔄 CACHE MISS — calling agent  key={_cache_key!r}")

    agent_graph = build_travel_agent()
    
    # Retry with exponential backoff on Groq 429 Rate Limit errors
    max_attempts = 4
    agent_result = {}
    for attempt in range(1, max_attempts + 1):
        try:
            agent_result = agent_graph.invoke({
                "messages": trimmed
            })
            break
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower() or "tokens" in err_str.lower()) and attempt < max_attempts:
                sleep_time = attempt * 2
                print(f"[Groq Rate Limit] Attempt {attempt}/{max_attempts} failed (429 Rate Limit). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise e

    messages = agent_result.get("messages", [])

    itinerary = ""
    if messages:
        last = messages[-1]
        itinerary = last.content if hasattr(last, "content") else str(last)

    flights     = []
    hotels      = []
    restaurants = []
    weather_str = ""
    budget_str  = ""
    route_str   = ""

    for msg in messages:
        if getattr(msg, "type", None) != "tool":
            continue
        tool_name = getattr(msg, "name", "")
        content   = getattr(msg, "content", "")

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                content = {}

        if not isinstance(content, dict):
            continue

        if tool_name == "search_flights":
            flights = content.get("flights", [])
        elif tool_name == "search_hotels":
            hotels = content.get("hotels", [])
        elif tool_name == "find_restaurants":
            restaurants = content.get("restaurants", [])
        elif tool_name == "get_weather":
            weather_str = content.get("weather", "")
        elif tool_name == "convert_budget":
            budget_str = content.get("budget", "")
        elif tool_name == "build_route":
            route_str = content.get("route", content.get("maps_url", ""))

    result = {
        "itinerary":   itinerary,
        "flights":     flights,
        "hotels":      hotels,
        "restaurants": restaurants,
        "weather":     weather_str,
        "budget":      budget_str,
        "route":       route_str,
    }

    if redis_cache is not None:
        _redis_client.set(_cache_key, json.dumps(result), ex=_CACHE_TTL)
        print(f"\n[Redis] 💾 CACHE WRITE — stored result in Redis")
        print(f"[Redis] 💾 KEY  : {_cache_key!r}")
        print(f"[Redis] 💾 TTL  : {_CACHE_TTL}s (1 hour)")
        print(f"[Redis] 💾 KEYS : {list(result.keys())}")
        print(f"[Redis] 💾 Flights stored  : {len(flights)}")
        print(f"[Redis] 💾 Hotels stored   : {len(hotels)}")
        print(f"[Redis] 💾 Itinerary chars : {len(itinerary)}")

        _confirm = _redis_client.get(_cache_key)
        print(f"[Redis] ✅ Confirmed readable from Redis: {bool(_confirm)}")

    return result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def make_graph() -> StateGraph:
    """
    Shape:  START ──► validate_dates ──► agent_node ──► END

    validate_dates uses interrupt() when the user provides a past date.
    Greetings and partial messages flow straight through to agent_node.
    The agent’s system prompt handles collecting all missing details
    conversationally before generating a travel plan.
    """
    builder = StateGraph(state_schema=TravelState, context_schema=TravelContext)
    builder.add_node("validate_dates", validate_dates)
    builder.add_node("agent_node",     agent_node)
    builder.add_edge(START,            "validate_dates")
    builder.add_edge("validate_dates", "agent_node")
    builder.add_edge("agent_node",     END)
    return builder


# Module-level MemorySaver  (one instance reused across requests)
_memory = MemorySaver()


def compile_graph():
    """
    Compile the graph with MemorySaver checkpointer.
    MemorySaver is required for interrupt() to persist state and surface
    GraphInterrupt to the caller. Redis LangGraph cache is layered on top
    when Redis is available.
    """
    graph = make_graph()
    if redis_cache is None:
        return graph.compile(checkpointer=_memory)
    return graph.compile(checkpointer=_memory, cache=redis_cache)
