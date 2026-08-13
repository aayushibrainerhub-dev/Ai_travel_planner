from typing import TypedDict

RATES_TO_USD = {
    "USD": 1.0,
    "INR": 84.0,
    "EUR": 0.92,
    "GBP": 0.79,
}


def convert_amount(amount: float, from_curr: str, to_curr: str) -> float:
    """Convert amount between supported currencies (USD, INR, EUR, GBP)."""
    from_curr = (from_curr or "USD").upper().strip()
    to_curr = (to_curr or "USD").upper().strip()
    if from_curr == to_curr or amount <= 0:
        return amount
    from_rate = RATES_TO_USD.get(from_curr, 1.0)
    to_rate = RATES_TO_USD.get(to_curr, 1.0)
    usd_amount = amount / from_rate
    converted = usd_amount * to_rate
    return round(converted, 2)


class TravelState(TypedDict, total=False):
    preferences: dict[str, str]


def convert_budget(state: TravelState) -> TravelState:
    budget = state.get("preferences", {}).get("budget", "1000")
    currency = state.get("preferences", {}).get("currency", "USD")
    return {
        "budget": f"Estimated budget of {budget} {currency}, including flights, hotels, and meals.",
    }
