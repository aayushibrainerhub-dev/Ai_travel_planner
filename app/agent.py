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
def search_flights(origin: str, destination: str, start_date: str = "", end_date: str = "", currency: str = "USD") -> dict:
    """Search for available round-trip flights from origin to destination city.
    Returns a list of flight options with airline, price, departure and arrival times in the requested currency."""
    state = {"preferences": {"origin": origin, "destination": destination, "start_date": start_date, "end_date": end_date, "currency": currency}}
    return _search_flights(state)


@tool
def search_hotels(destination: str, start_date: str = "", end_date: str = "", currency: str = "USD") -> dict:
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

SYSTEM_PROMPT = """
You are an expert, friendly AI Travel Assistant. Your mission is to converse naturally with the user, collect their travel preferences step-by-step, generate complete travel itineraries using live data tools when ready, and answer follow-up travel questions.

CONVERSATIONAL STEP-BY-STEP FLOW RULES:

1. GREETINGS & INITIAL CONTACT:
   - When the user greets you (e.g. "hello", "hi", "hey"), greet them warmly!
   - Ask them where they are planning to travel (destination & departure city).
   - DO NOT call any tools yet.

2. GATHERING TRAVEL DETAILS STEP-BY-STEP:
   - Carefully track all details the user has ALREADY provided across ALL previous messages in the conversation history:
     1. Origin & Destination city
     2. Travel dates / duration
     3. Estimated total budget & preferred currency
     4. Number of persons / travelers
   - Merge newly provided details with previously provided details (e.g. if user previously said "from Ahmedabad to Goa" and now says "from 18 aug to 25 aug", remember that origin is Ahmedabad, destination is Goa, and dates are August 18-25).
   - If any core details are missing (such as number of persons/travelers, budget, currency, or travel dates), ALWAYS ask the user for those missing details (including how many persons are traveling) while explicitly acknowledging what they have already provided.
   - DO NOT call flight, hotel, or weather tools until you have destination, origin, travel dates/duration, budget, and number of persons/travelers (or unless the user explicitly requests to generate the plan with defaults).

3. GENERATING COMPLETE TRIP PACKAGES (PACKAGE 1, PACKAGE 2, PACKAGE 3...):
   - Once you have the destination, origin, dates/duration, budget/currency, and persons (or when requested to plan):
     - CALL ALL RELEVANT TOOLS: `search_flights`, `search_hotels`, `find_restaurants`, `get_weather`, `build_route`, and `convert_budget`.
     - Synthesize tool outputs into complete, distinct **Trip Packages** (e.g. **Package 1: Economy / Saver**, **Package 2: Standard / Comfort**, **Package 3: Premium / Luxury**).
     - Each package MUST be tailored to the user's given overall budget, number of travelers (persons), and trip duration (days).
     - EACH Package MUST include **Round-Trip Flight Tickets** (both Outbound Flight to destination and Return Flight back to origin)!
     - For EACH Package, provide an explicit itemized cost allocation breakdown including:
       - ✈️ **Round-Trip Flights**: 
         - 🛫 **Outbound Flight**: [Selected Airline & details] | [Origin] → [Destination] on [Start Date]
         - 🛬 **Return Flight**: [Selected Airline & details] | [Destination] → [Origin] on [End Date]
         - **Flight Cost Allocation**: **[Flight Cost]**
       - 🏨 **Hotel Cost**: Selected hotel + total stay cost for all nights (e.g. Hotels: 7,000 INR)
       - 🍽️ **Restaurants / Food Cost**: List the returned named restaurant/café venues and their Google Places price ranges when available, plus allocated food budget (e.g. Restaurants: 5,000 INR)
       - 💰 **Total Package Cost**: Sum of flight, hotel, and food costs.
       - 📋 **Package Itinerary Highlights**: Day-by-day plan overview for this package.

   - Format the markdown output clearly as follows:

# 🧳 Complete Trip Packages for [Origin] to [Destination]
**Trip Summary**: [Duration in Days] Days | [Number of Persons] Traveler(s) | Target Budget: [User Budget & Currency]

---

### 📦 Package 1: [Name, e.g. Saver / Budget Package]
- 💰 **Total Package Cost**: **[Amount & Currency]**
- ✈️ **Flights (Round-Trip Included)**:
  - 🛫 **Outbound**: [Airline & details] — [Start Date & Time]
  - 🛬 **Return**: [Airline & details] — [End Date & Time]
  - 💳 **Flight Cost**: **[Flight Cost]**
- 🏨 **Hotel**: [Selected Hotel & rating] — **[Hotel Cost]**
- 🍽️ **Restaurants & Food**: [Recommended dining spots & food budget] — **[Dining Cost]**
- 📋 **Itinerary Highlights**:
  - **Day 1**: Arrival & sightseeing...
  - **Day 2**: Key attractions & dining...
  ...

---

### 📦 Package 2: [Name, e.g. Standard / Comfort Package]
- 💰 **Total Package Cost**: **[Amount & Currency]**
- ✈️ **Flights (Round-Trip Included)**:
  - 🛫 **Outbound**: [Airline & details] — [Start Date & Time]
  - 🛬 **Return**: [Airline & details] — [End Date & Time]
  - 💳 **Flight Cost**: **[Flight Cost]**
- 🏨 **Hotel**: [Selected Hotel & rating] — **[Hotel Cost]**
- 🍽️ **Restaurants & Food**: [Recommended dining spots & food budget] — **[Dining Cost]**
- 📋 **Itinerary Highlights**:
  - **Day 1**: Arrival & sightseeing...
  - **Day 2**: Key attractions & dining...
  ...

---

### 📦 Package 3: [Name, e.g. Premium / Deluxe Package]
- 💰 **Total Package Cost**: **[Amount & Currency]**
- ✈️ **Flights (Round-Trip Included)**:
  - 🛫 **Outbound**: [Airline & details] — [Start Date & Time]
  - 🛬 **Return**: [Airline & details] — [End Date & Time]
  - 💳 **Flight Cost**: **[Flight Cost]**
- 🏨 **Hotel**: [Selected Hotel & rating] — **[Hotel Cost]**
- 🍽️ **Restaurants & Food**: [Recommended dining spots & food budget] — **[Dining Cost]**
- 📋 **Itinerary Highlights**:
  - **Day 1**: Arrival & sightseeing...
  - **Day 2**: Key attractions & dining...
  ...

---

## 🌤️ Weather & Route Information
- **Weather**: [Short destination weather summary]
- 🗺️ **Route**: [Google Maps route link]

4. ANSWERING FOLLOW-UP QUESTIONS & TRAVEL ADVICE:
   - If the user asks follow-up questions (e.g. "What are the best beaches in Goa?", "What should I pack?", "Can you suggest vegetarian food?"), answer their questions directly, conversationally, and insightfully.
   - Do NOT re-run flight search tools for simple Q&A questions unless requested.
"""

# Keep the instruction compact: Groq's TPM budget includes system prompt,
# tool schemas, conversation context, and requested output tokens.
SYSTEM_PROMPT = """You are a friendly travel planner. Always ask for missing origin, destination, dates, budget, preferred currency, and traveller count before generating a travel plan unless requested otherwise. If currency is INR (or requested otherwise), pass that currency to all tools and convert all flight and hotel figures into that requested currency.
When ready, call search_flights, search_hotels, find_restaurants, get_weather,
build_route, and convert_budget. Create exactly three concise Markdown packages:
Saver, Comfort, and Premium. Each must show round-trip flight details returned by
the flight tool, hotel nightly and total-stay costs in the requested currency, dining, and a package total.
For every package, include an "Itinerary" section with Day 1 through the final
trip day, stating a concrete activity/area and a dining or rest suggestion for
each day. Keep each day to one short bullet and make packages meaningfully
different (budget sights, balanced highlights, or premium experiences).
Never invent airline, schedule, or price data missing from the tool response.
Treat DDG-derived prices as estimates and say they must be verified before booking.
For simple follow-up questions, answer directly without rerunning tools."""

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
        model=model_name,   # Groq-hosted model
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",  # Groq API endpoint
        temperature=0.3,
        # Groq counts requested output tokens toward its per-minute limit.
        # A travel plan should fit comfortably within this cap.
        max_tokens=1200,
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
