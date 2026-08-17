from typing import List
from langgraph.types import Command
from app.graph.builder import compile_graph, TravelContext


def run_travel_graph(
    user_id: str,
    preferences: dict,
    messages: List[dict] | None = None,
) -> dict:
    graph = compile_graph()
    context = TravelContext(user_id=user_id)
    config = {"configurable": {"thread_id": user_id}}

    # Check if the graph is currently paused at an interrupt for this thread.
    # If so, resume it (passing updated state) instead of re-invoking from START.
    state = graph.get_state(config)
    is_interrupted = bool(state and state.next and state.tasks and
                          any(t.interrupts for t in state.tasks))

    if is_interrupted:
        # Patch the checkpointed state with the latest preferences & messages
        # so validate_dates re-runs against the new dates, not the old ones.
        graph.update_state(config, {"preferences": preferences, "messages": messages or []})
        result = graph.invoke(
            Command(resume=None),
            config=config,
            context=context,
        )
    else:
        result = graph.invoke(
            {"preferences": preferences, "messages": messages or []},
            config=config,
            context=context,
        )
    return result
