import json
import re
import datetime
from pathlib import Path

MEMORY_PATH = Path("data/operator_memory.json")
USER_PREFS_PATH = Path("data/user_preferences.json")
SESSION_MEMORY_PATH = Path("data/session_memory.json")

class MemoryManager:
    def __init__(self):
        self.operator_memory = self._load_operator_memory()
        self.user_preferences = self._load_user_preferences()
        self.session_memory = self._load_session_memory()

    @staticmethod
    def _timestamp_now():
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _load_operator_memory(self):
        default_memory = {"operator_name": "", "notes": [], "personal_facts": []}
        if not MEMORY_PATH.exists():
            return default_memory
        try:
            with MEMORY_PATH.open("r", encoding="utf-8") as file_handle:
                loaded = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_memory
        if not isinstance(loaded, dict):
            return default_memory
        notes = loaded.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        personal_facts = loaded.get("personal_facts", [])
        if not isinstance(personal_facts, list):
            personal_facts = []
        return {
            "operator_name": str(loaded.get("operator_name", "")).strip(),
            "notes": [str(note).strip() for note in notes if str(note).strip()],
            "personal_facts": [str(fact).strip() for fact in personal_facts if str(fact).strip()],
        }

    def _save_operator_memory(self):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.operator_memory, file_handle, indent=2)

    def _load_user_preferences(self):
        default_data = {"preferences": []}
        if not USER_PREFS_PATH.exists():
            return default_data
        try:
            with USER_PREFS_PATH.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_data
        prefs = payload.get("preferences", []) if isinstance(payload, dict) else []
        if not isinstance(prefs, list):
            prefs = []
        cleaned = [str(item).strip() for item in prefs if str(item).strip()]
        return {"preferences": cleaned[-100:]}

    def _save_user_preferences(self):
        USER_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with USER_PREFS_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.user_preferences, file_handle, indent=2)

    def _load_session_memory(self):
        default_data = {"recent_turns": [], "current_intent": "general", "facts_cache": []}
        if not SESSION_MEMORY_PATH.exists():
            return default_data
        try:
            with SESSION_MEMORY_PATH.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_data
        if not isinstance(payload, dict):
            return default_data
        turns = payload.get("recent_turns", [])
        facts = payload.get("facts_cache", [])
        if not isinstance(turns, list):
            turns = []
        if not isinstance(facts, list):
            facts = []
        return {
            "recent_turns": turns[-25:],
            "current_intent": str(payload.get("current_intent", "general")).strip() or "general",
            "facts_cache": [str(item).strip() for item in facts if str(item).strip()][-25:],
        }

    def _save_session_memory(self):
        SESSION_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SESSION_MEMORY_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.session_memory, file_handle, indent=2)

    @staticmethod
    def _extract_preference_statement(text):
        lowered = str(text or "").strip().lower()
        match = re.search(r"\b(?:i prefer|my preference is|please always)\s+(.+)$", lowered)
        if not match:
            return ""
        return match.group(1).strip(" .")

    def _capture_preference(self, text):
        preference = self._extract_preference_statement(text)
        if not preference:
            return
        prefs = self.user_preferences.setdefault("preferences", [])
        if preference not in prefs:
            prefs.append(preference)
        self.user_preferences["preferences"] = prefs[-100:]
        self._save_user_preferences()

    def _update_session_memory(self, user_text, assistant_text, intent="general", facts=None):
        turns = self.session_memory.setdefault("recent_turns", [])
        turns.append(
            {
                "timestamp": self._timestamp_now(),
                "user": str(user_text or "").strip(),
                "assistant": str(assistant_text or "").strip(),
            }
        )
        self.session_memory["recent_turns"] = turns[-25:]
        self.session_memory["current_intent"] = intent or "general"
        if facts:
            cache = self.session_memory.setdefault("facts_cache", [])
            cache.extend([str(item).strip() for item in facts if str(item).strip()])
            self.session_memory["facts_cache"] = cache[-25:]
        self._save_session_memory()
