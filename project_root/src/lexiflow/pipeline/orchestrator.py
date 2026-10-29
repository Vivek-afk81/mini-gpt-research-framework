"""End-to-end LexiFlow orchestration."""

from .drafter import draft
from .memory import Memory
from .retriever import PolicyRetriever
from .translator import Translator


class LexiFlow:
    def __init__(self, dataset_path="data/campus_messages_v2.jsonl", policy_dir="data/policies", memory_path="lexiflow.db", backend="local"):
        self.backend = backend
        self.translator = Translator(dataset_path=dataset_path, backend=backend)
        self.retriever = PolicyRetriever(policy_dir=policy_dir)
        self.memory = Memory(memory_path)

    def run(self, user_id, hinglish):
        translation = self.translator.translate(hinglish)
        sources = self.retriever.search(translation["english"], k=3)
        profile = self.memory.profile(user_id)
        past = self.memory.recent(user_id, category=translation["category"])
        email = draft(translation["english"], translation["category"], translation["tone"], sources, profile, self.backend)
        self.memory.log_request(user_id, hinglish, translation["english"], translation["category"], translation["tone"], email)
        return {**translation, "sources": sources, "draft": email, "past": past}