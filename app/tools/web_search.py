"""Small, best-effort DuckDuckGo search utility for travel context."""

import re


_PRICE_PATTERN = re.compile(
    r"(?:"
    r"(?P<symbol>₹|\$|€|£)\s*(?P<symbol_amount>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"|(?P<code_amount>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?P<code>INR|USD|EUR|GBP)\b"
    r"|(?P<prefix_code>INR|USD|EUR|GBP)\s*(?P<prefix_amount>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r")",
    re.IGNORECASE,
)
_CURRENCY_BY_SYMBOL = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
_PLAIN_PRICE_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b|\b\d{3,}(?:\.\d{1,2})?\b")


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Return a compact, serialisable set of DuckDuckGo text-search results.

    Search results are supporting context only.  They must not be treated as
    confirmed inventory, schedules, or prices.
    """
    try:
        from ddgs import DDGS

        results = list(DDGS().text(
            query,
            region="in-en",
            safesearch="moderate",
            max_results=max_results,
        ))
    except Exception as exc:
        # Web search should never prevent the primary travel provider from
        # returning its results.
        print(f"DuckDuckGo search unavailable ({exc}).")
        return []

    details: list[dict[str, str]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        details.append(
            {
                "title": str(result.get("title", "")).strip()[:160],
                "url": str(result.get("href", result.get("url", ""))).strip(),
                "snippet": str(result.get("body", result.get("snippet", ""))).strip()[:280],
            }
        )
    return [detail for detail in details if detail["title"] or detail["snippet"]]


def travel_web_details(query: str, max_results: int = 1) -> list[dict[str, str]]:
    """Return compact DDG results; callers can request more for price lookup."""
    return search_web(query, max_results=max_results)


def extract_web_price(
    web_details: list[dict[str, str]],
    *,
    entity_name: str = "",
    required_context: tuple[str, ...] = (),
    default_currency: str = "",
) -> dict[str, str | float] | None:
    """Extract the first explicitly currency-qualified amount from a DDG snippet.

    The result deliberately contains the source URL and original amount text so
    callers can label it as a web estimate instead of a provider-confirmed rate.
    """
    expected_name = " ".join(entity_name.lower().split())
    print(f"[DDG Price] Evaluating {len(web_details)} result(s) for: {entity_name or 'unspecified item'}")
    for detail in web_details:
        title = detail.get("title", "")
        snippet = detail.get("snippet", "")
        combined = f"{title} {snippet}".lower()
        print(f"[DDG Price] Result title={title!r} snippet={snippet!r}")
        if expected_name and expected_name not in combined:
            print(f"[DDG Price] Hotel name not present; checking this result as a fallback price candidate.")

        matched_amount = False
        for match in _PRICE_PATTERN.finditer(snippet):
            matched_amount = True
            nearby_text = snippet[max(0, match.start() - 80):match.end() + 80].lower()
            if required_context and not any(term.lower() in nearby_text for term in required_context):
                print(f"[DDG Price] Rejected {match.group(0)!r}: missing context {required_context}.")
                continue
            amount_text = (
                match.group("symbol_amount")
                or match.group("code_amount")
                or match.group("prefix_amount")
            )
            try:
                amount = float(amount_text.replace(",", ""))
            except (AttributeError, ValueError):
                print(f"[DDG Price] Rejected {match.group(0)!r}: amount could not be parsed.")
                continue
            currency = (
                _CURRENCY_BY_SYMBOL.get(match.group("symbol"), "")
                or (match.group("code") or match.group("prefix_code") or "").upper()
            )
            if currency:
                price_kind = (
                    "total_stay"
                    if any(term in nearby_text for term in ("total", "total stay", "package price", "package", "for 7 persons", "for 6 persons", "for 5 persons", "for 4 persons", "for 3 persons", "for 2 persons", "entire stay"))
                    else "per_night" if any(term in nearby_text for term in ("per night", "/night", "nightly", "a night", "per-night"))
                    else "unspecified"
                )
                print(f"[DDG Price] Accepted {currency} {amount:,.2f} ({price_kind}) from {detail.get('url', '')!r}.")
                return {
                    "amount": amount,
                    "currency": currency,
                    "text": match.group(0),
                    "url": detail.get("url", ""),
                    "price_kind": price_kind,
                }
            print(f"[DDG Price] Rejected {match.group(0)!r}: no recognised currency.")
        if not matched_amount:
            # DDG snippets frequently omit the currency symbol. For hotel
            # pricing, accept a plain number only if nearby text indicates a price.
            for plain_match in _PLAIN_PRICE_PATTERN.finditer(snippet):
                match_str = plain_match.group(0)
                nearby_text = snippet[max(0, plain_match.start() - 60):plain_match.end() + 60].lower()
                if any(bad in nearby_text for bad in ("phone", "call", "pin", "zip", "code", "pincode", "tel", "contact", "review", "id", "ref")):
                    continue
                if not any(good in nearby_text for good in ("price", "rate", "cost", "night", "room", "rs", "inr", "usd", "eur", "gbp", "stay", "total", "fee")):
                    continue
                if default_currency:
                    try:
                        amount = float(match_str.replace(",", ""))
                    except ValueError:
                        continue
                    price_kind = (
                        "total_stay"
                        if any(term in nearby_text for term in ("total", "total stay", "package", "entire stay"))
                        else "per_night" if any(term in nearby_text for term in ("per night", "/night", "nightly", "a night"))
                        else "unspecified"
                    )
                    print(
                        f"[DDG Price] Accepted {default_currency.upper()} {amount:,.2f} ({price_kind}) from plain "
                        f"DDG amount {match_str!r}; currency inferred from trip preference."
                    )
                    return {
                        "amount": amount,
                        "currency": default_currency.upper(),
                        "text": match_str,
                        "url": detail.get("url", ""),
                        "currency_inferred": True,
                        "price_kind": price_kind,
                    }
            print("[DDG Price] This result contains no numeric price amount.")
    print("[DDG Price] DDG returned no numeric price in any searched snippet; returning None.")
    return None
