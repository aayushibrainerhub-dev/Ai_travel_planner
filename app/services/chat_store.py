"""
chat_store.py — In-memory chat session store using LangChain's InMemoryStore.

Stores per-user session data:
  - "messages": list of message dicts [{"role": "user"|"assistant", "content": str}]
  - "preferences": dict of extracted preferences (origin, destination, dates, budget, currency)
"""

from typing import Dict, List, Optional
try:
    from langchain_core.stores import InMemoryStore as LangChainInMemoryStore
except ImportError:
    from langchain.stores import InMemoryStore as LangChainInMemoryStore


class ChatStoreManager:
    """Session store manager using LangChain's InMemoryStore."""

    def __init__(self) -> None:
        self.store = LangChainInMemoryStore()

    def get_session(self, user_id: str) -> dict:
        """Retrieve full session object for a given user_id."""
        items = self.store.mget([user_id])
        if items and items[0] is not None:
            return items[0]
        return {"messages": [], "preferences": {}}

    def get_history(self, user_id: str) -> List[dict]:
        """Return conversation message history for a user."""
        session = self.get_session(user_id)
        return list(session.get("messages", []))

    def get_preferences(self, user_id: str) -> dict:
        """Return stored preferences for a user."""
        session = self.get_session(user_id)
        return dict(session.get("preferences", {}))

    def save_session(self, user_id: str, messages: List[dict], preferences: dict) -> None:
        """Save updated messages and preferences to InMemoryStore."""
        self.store.mset([(user_id, {"messages": messages, "preferences": preferences})])

    def append(self, user_id: str, role: str, content: str) -> None:
        """Append a message to a user's session in InMemoryStore."""
        session = self.get_session(user_id)
        messages = session.get("messages", [])
        messages.append({"role": role, "content": content})
        self.store.mset([(user_id, {"messages": messages, "preferences": session.get("preferences", {})})])

    def clear(self, user_id: str) -> None:
        """Clear conversation session for a user."""
        self.store.mdelete([user_id])

    def clear_all(self) -> None:
        """Wipe all sessions by resetting the store."""
        self.store = LangChainInMemoryStore()


# Singleton instance imported across the app
chat_store = ChatStoreManager()

