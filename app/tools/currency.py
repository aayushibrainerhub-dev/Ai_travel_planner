from typing import TypedDict


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]


def convert_budget(state: TravelState) -> TravelState:
    budget = state.get("preferences", {}).get("budget", "1000")
    currency = state.get("preferences", {}).get("currency", "USD")
    return {
        "budget": f"Estimated budget of {budget} {currency}, including flights, hotels, and meals.",
    }
