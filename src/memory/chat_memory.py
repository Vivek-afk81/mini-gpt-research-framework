"""
LexiFlow — Conversation memory.

Maintains per-session chat history with a configurable window size.
Formats history for injection into LLM prompts.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class Message:
    """A single message in the conversation."""
    role: str           # "user" or "assistant"
    content: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ChatMemory:
    """
    Manages conversation history for a single session.

    Keeps the last `max_turns` exchanges (user + assistant pairs)
    to stay within LLM context limits.
    """

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add_user_message(self, content):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content):
        """Add an assistant response to history."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self):
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self):
        """Return the current message history."""
        return list(self.messages)

    def format_for_prompt(self):
        """
        Format conversation history as a string for LLM context.

        Example output:
            User: What is the attendance requirement?
            Assistant: The minimum attendance requirement is 75%.
            User: What happens if I have less than that?
        """
        if not self.messages:
            return ""

        lines = []
        for msg in self.messages:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)

    def format_as_chat_messages(self):
        """
        Format history as a list of dicts for chat-style LLM APIs.

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self):
        """Clear all conversation history."""
        self.messages.clear()

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return f"ChatMemory(turns={len(self.messages) // 2}, messages={len(self.messages)})"
