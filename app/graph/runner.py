from typing import List
from app.graph.builder import compile_graph, TravelContext


def run_travel_graph(
    user_id: str,
    preferences: dict,
    messages: List[dict] | None = None,
) -> dict:
    graph = compile_graph()
    context = TravelContext(user_id=user_id)
    result = graph.invoke(
        {"preferences": preferences, "messages": messages or []},
        context=context,
    )
    return result
