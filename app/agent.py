"""
agent.py — LangChain tool-calling agent (langchain >= 1.0 API)

In langchain 1.x, `create_tool_calling_agent` was replaced by `create_agent`.
`create_agent` is even better for our use-case because it returns a native
LangGraph CompiledStateGraph that runs the tool-calling loop internally.

Usage:
    graph = build_travel_agent()
    result = graph.invoke({"messages": [{"role": "user", "content": "Plan a trip ..."}]})
    final_message = result["messages"][-1].content
"""

import os
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Grok API key loader (reads .env file when env var is not set by the shell)
# ---------------------------------------------------------------------------

def _load_grok_api_key() -> str:
    """Read GROK_API_CLOUD_KEY from env var or fall back to .env file."""
    api_key = os.getenv("GROK_API_CLOUD_KEY", "").strip().strip('"').strip("'")
    if api_key:
        return api_key

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if "=" in stripped and stripped.split("=", 1)[0].strip() == "GROK_API_CLOUD_KEY":
                _, value = stripped.split("=", 1)
                return value.strip().strip('"').strip("'")
    return ""


# Import the original state-based tool functions from the project
from app.tools.flight import search_flights as _search_flights
from app.tools.hotel import search_hotels as _search_hotels
from app.tools.restaurant import find_restaurants as _find_restaurants
from app.tools.weather import get_weather as _get_weather
from app.tools.maps import build_route as _build_route
from app.tools.currency import convert_budget as _convert_budget


# ---------------------------------------------------------------------------
# Wrap state-based functions into plain LangChain tools
# (create_agent accepts any callable with a docstring)
# ---------------------------------------------------------------------------

@tool
def search_flights(origin: str, destination: str, start_date: str, end_date: str, currency: str = "USD") -> dict:
    """Search for available round-trip flights from origin to destination city.
    Returns a list of flight options with airline, price, departure and arrival times in the requested currency."""
    state = {"preferences": {"origin": origin, "destination": destination, "start_date": start_date, "end_date": end_date, "currency": currency}}
    return _search_flights(state)


@tool
def search_hotels(destination: str, start_date: str, end_date: str, currency: str = "USD") -> dict:
    """Search for hotel accommodations in the given destination city.
    Include the trip start and end dates to calculate the number of nights and total hotel stay price.
    Returns a list of hotels with per-night and total-stay prices when DDG provides an explicit price.
    Do not treat a Places price range as a confirmed nightly room price."""
    state = {"preferences": {"destination": destination, "start_date": start_date, "end_date": end_date, "currency": currency}}
    return _search_hotels(state)


@tool
def find_restaurants(destination: str) -> dict:
    """Find named restaurants and cafés in the given destination city.
    Returns venue names, cuisine/type, rating, address, and Google Places price range/level when available."""
    state = {"preferences": {"destination": destination}}
    return _find_restaurants(state)


@tool
def get_weather(destination: str) -> dict:
    """Get current weather information for the destination city.
    Returns a weather summary string."""
    state = {"preferences": {"destination": destination}}
    return _get_weather(state)


@tool
def build_route(origin: str, destination: str) -> dict:
    """Build a Google Maps route URL from origin to destination.
    Returns a maps URL for the route."""
    state = {"preferences": {"origin": origin, "destination": destination}}
    return _build_route(state)


@tool
def convert_budget(budget: str, currency: str = "EUR") -> dict:
    """Convert and format the travel budget.
    Returns the formatted budget string."""
    state = {"preferences": {"budget": budget, "currency": currency}}
    return _convert_budget(state)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly travel planner.
All required details (origin, destination, start date, end date, budget, traveller count) are already collected and provided in the system context message. Call all tools immediately: search_flights, search_hotels, find_restaurants, get_weather, build_route, convert_budget.
If currency is INR pass it to all tools. Never invent prices — use tool results only.
Output exactly three Markdown packages: Saver, Comfort, Premium. Each: round-trip flights, hotel total, dining budget, package total, day-by-day itinerary (one short bullet per day). Keep output concise."""

TRAVEL_TOOLS = [
    search_flights,
    search_hotels,
    find_restaurants,
    get_weather,
    build_route,
    convert_budget,
]


def build_travel_agent():
    """
    Build and return a LangGraph CompiledStateGraph powered by
    LangChain's create_agent (langchain 1.x API).

    The returned graph accepts:
        {"messages": [{"role": "user", "content": "<user query>"}]}

    And returns a state dict with `messages`, where the last message
    is the agent's final answer.
    """
    api_key = _load_grok_api_key()
    if not api_key:
        raise ValueError(
            "GROK_API_CLOUD_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3,
        max_tokens=800,
        max_retries=5,
    )

    # create_agent returns a CompiledStateGraph with built-in tool loop
    agent_graph = create_agent(
        model=llm,
        tools=TRAVEL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent_graph


#mock demo
if __name__ == "__main__":
    agent = build_travel_agent()
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Plan a 5-day trip from London to Paris starting 2026-08-15 with a budget of 1500 EUR.",
        }]
    })
    print(result["messages"][-1].content)
