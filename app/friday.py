import queue
import sys
import os

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_service import load_dataset
from models.RAG.brain import Brain
from models.RAG.generator import LocalInstructionGenerator
import ast
import datetime
import difflib
import html
import json
import operator
import speech_recognition as sr

import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

import pyttsx3

try:
    import speech_recognition as sr
except Exception:
    sr = None

from core.inference import InferenceManager, OFFLINE_CAPITALS, OFFLINE_PEOPLE, OFFLINE_FACTS
from models.finetuning.generate import generate_text, reload_generator


WAKE_WORDS = ('friday', 'hey friday')
WAKE_ALIASES = ('phriday', 'fry day', 'fri day')
EXIT_WORDS = ("quit", "exit", "goodbye", "shutdown")
ACTIVE_WINDOW_SECONDS = 14
MEMORY_PATH = Path("data/operator_memory.json")
TASKS_PATH = Path("data/daily_tasks.json")
LEARNED_QA_PATH = Path("data/learned_qa.txt")
LEARNED_QA_META_PATH = Path("data/learned_qa.json")
CORPUS_PATH = Path("data/corpus.txt")
USER_PREFS_PATH = Path("data/user_preferences.json")
SESSION_MEMORY_PATH = Path("data/session_memory.json")
ONLINE_STT_MAX_FAILURES = 3
DUPLICATE_UTTERANCE_WINDOW_SECONDS = 2.2

APP_OPEN_COMMANDS = {
    "notepad": ("notepad", "Opening Notepad."),
    "calculator": ("calc", "Opening Calculator."),
    "command prompt": ("cmd", "Opening Command Prompt."),
    "file explorer": ("explorer", "Opening File Explorer."),
}

APP_CLOSE_IMAGES = {
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",
    "command prompt": "cmd.exe",
}







class FridayAssistant:

    def scan_perimeter(self):
        print("Scanning house perimeter... All sensors are nominal. Perimeter is secure.")
        
    def lock_doors(self):
        print("Engaging all household locks and activating security protocols. The house is now protected.")
        
    def security_report(self):
        print("Conducting full security diagnostic. Cameras, motion sensors, and alarms are fully operational.")
    def __init__(self, samplerate=16000):
        self.engine = pyttsx3.init()
        self._configure_voice()

        self.samplerate = samplerate
        self.recognizer = sr.Recognizer() if sr is not None else None
        
        self.audio_queue = queue.Queue()
        self.online_stt_enabled = True
        self.online_stt_failures = 0

        dataset = load_dataset()
        self.brain = Brain(dataset)
        self.rag_generator = LocalInstructionGenerator()
        self.qa_lookup = self._build_local_qa_lookup(dataset)
        self.qa_questions = list(self.qa_lookup.keys())
        

        self.sleep_mode = False
        self.history = []
        self.active_until = 0.0
        self.operator_memory = self._load_operator_memory()
        self.user_preferences = self._load_user_preferences()
        self.daily_planner = self._load_daily_planner()
        self.session_memory = self._load_session_memory()
        self.conversational_mode = False
        self.memory_tiers = {
            "short_term": self.session_memory,
            "long_term_preferences": self.user_preferences,
            "task_memory": self.daily_planner,
        }
        self.conversation_state = {
            "last_topic": "",
            "last_user_text": "",
            "last_assistant_text": "",
            "turn_count": 0,
            "last_user_mood": "neutral",
            "last_goal": "",
            "last_knowledge_intent": "",
            "last_knowledge_subject": "",
            "last_plan_goal": "",
            "last_plan_steps": [],
        }
        self.knowledge_cache = {}
        self.inference_manager = InferenceManager(self)
        self.active_timers = []
        self._last_reminder_tick = ""
        self._last_handled_text = ""
        self._last_handled_at = 0.0

    @staticmethod
    def _compact_answer(text, max_chars=280):
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return ""

        # A non-positive limit means "do not truncate".
        if max_chars is None or max_chars <= 0:
            return cleaned

        if len(cleaned) <= max_chars:
            return cleaned

        cutoff = cleaned.rfind(" ", 0, max_chars)
        if cutoff <= 0:
            cutoff = max_chars
        return cleaned[:cutoff].strip() + "..."

    @staticmethod
    def _clean_corpus_line(text):
        cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", str(text or "").strip().lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _is_question(text):
        normalized = text.strip().lower()
        if not normalized:
            return False
        if normalized.endswith("?"):
            return True

        starters = (
            "who ",
            "what ",
            "when ",
            "where ",
            "why ",
            "how ",
            "which ",
            "whom ",
            "whose ",
            "convert ",
            "calculate ",
            "compute ",
            "solve ",
            "is ",
            "are ",
            "can ",
            "could ",
            "should ",
            "would ",
            "do ",
            "does ",
            "did ",
            "will ",
            "explain ",
            "tell me ",
        )
        return normalized.startswith(starters)

    def _normalize_question_query(self, text):
        query = text.strip().lower().strip(" ?")
        query = re.sub(
            r"^(?:friday\s+)?(?:please\s+)?(?:tell me|explain|define|what is|who is|where is|when is|why is|how does|how do|how can)\s+",
            "",
            query,
        )
        return query.strip() or text.strip()

    def _normalize_qa_key(self, text):
        key = self._normalize_key(text)
        key = key.strip(" ?!.,")
        key = re.sub(r"\bwhat's\b|\bwhats\b", "what is", key)
        key = re.sub(r"\bwho's\b", "who is", key)
        key = re.sub(r"\bi'm\b", "i am", key)
        key = re.sub(r"\bcan u\b", "can you", key)
        key = re.sub(r"\bpls\b", "please", key)
        key = re.sub(
            r"^(?:friday\s+)?(?:please\s+)?(?:can you tell me|could you tell me|can you|could you|would you|do you know)\s+",
            "",
            key,
        )
        key = re.sub(r"[^a-z0-9\s]", " ", key)
        key = re.sub(r"\s+", " ", key).strip()
        return key

    def _build_local_qa_lookup(self, dataset):
        lookup = {}
        for item in dataset:
            if isinstance(item, dict):
                question = item.get("question", "")
                answer = item.get("answer", "")
            else:
                question, answer = item
            normalized = self._normalize_qa_key(question)
            answer_text = str(answer).strip()
            if normalized and answer_text:
                lookup[normalized] = answer_text

        # Normalized aliases for common phrasing variants.
        if "what is your name" in lookup:
            lookup.setdefault("your name", lookup["what is your name"])
            lookup.setdefault("whats your name", lookup["what is your name"])
        if "who are you" in lookup:
            lookup.setdefault("tell me about yourself", lookup["who are you"])
        return lookup

    def _reload_nlp_models(self):
        dataset = load_dataset()
        self.brain = Brain(dataset)
        self.qa_lookup = self._build_local_qa_lookup(dataset)
        self.qa_questions = list(self.qa_lookup.keys())

    def _persist_qa_pair(self, question, answer):
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        if not question or not answer:
            return False

        entries = {}
        if LEARNED_QA_PATH.exists():
            try:
                with LEARNED_QA_PATH.open("r", encoding="utf-8") as file_handle:
                    for raw_line in file_handle:
                        line = raw_line.strip()
                        if not line or "|||" not in line:
                            continue
                        old_question, old_answer = line.split("|||", 1)
                        key = self._normalize_qa_key(old_question)
                        if key:
                            entries[key] = (old_question.strip(), old_answer.strip())
            except OSError:
                return False

        meta_entries = {}
        if LEARNED_QA_META_PATH.exists():
            try:
                with LEARNED_QA_META_PATH.open("r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        old_question = str(item.get("question", "")).strip()
                        old_answer = str(item.get("answer", "")).strip()
                        metadata = item.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                        key = self._normalize_qa_key(old_question)
                        if key and old_question and old_answer:
                            meta_entries[key] = {
                                "question": old_question,
                                "answer": old_answer,
                                "metadata": metadata,
                            }
            except (OSError, json.JSONDecodeError):
                return False

        key = self._normalize_qa_key(question)
        if not key:
            return False
        entries[key] = (question, answer)
        meta_entries[key] = {
            "question": question,
            "answer": answer,
            "metadata": {
                "source": "user_teach",
                "confidence": 1.0,
                "timestamp": self._timestamp_now(),
                "topic": self.inference_manager._extract_focus_phrase(question) or "general",
            },
        }

        LEARNED_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEARNED_QA_PATH.open("w", encoding="utf-8") as file_handle:
            for stored_question, stored_answer in entries.values():
                file_handle.write(f"{stored_question}|||{stored_answer}\n")

        LEARNED_QA_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEARNED_QA_META_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(list(meta_entries.values()), file_handle, indent=2)
        return True

    def _record_interaction_for_training(self, user_text, assistant_text):
        user_line = self._clean_corpus_line(user_text)
        assistant_line = self._clean_corpus_line(assistant_text)
        fragments = [part for part in (user_line, assistant_line) if part]
        if not fragments:
            return

        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS_PATH.open("a", encoding="utf-8") as file_handle:
            file_handle.write(" ".join(fragments) + "\n")

    def _teach_nlp_pair(self, text):
        qa_pattern = re.search(
            r"\b(?:learn qa|teach qa|learn this|teach this)\s+(.+?)\s*(?:\|\|\||=>|->)\s*(.+)$",
            text,
        )
        if not qa_pattern:
            return None

        question = qa_pattern.group(1).strip(" .")
        answer = qa_pattern.group(2).strip()
        if not question or not answer:
            return "Please provide both question and answer."

        if len(question.split()) < 2:
            return "Use a clearer question so I can learn it correctly."

        if len(answer) < 3:
            return "Please provide a fuller answer for learning."

        persisted = self._persist_qa_pair(question, answer)
        if not persisted:
            return "I could not store that learning item."

        self._reload_nlp_models()
        self._record_interaction_for_training(question, answer)
        return "Learned and indexed. I can answer that from now on."



    def _detect_intent(self, text):
        lowered = str(text or "").lower()
        if any(token in lowered for token in ("task", "todo", "plan", "reminder", "timer")):
            return "task"
        if any(token in lowered for token in ("who", "what", "when", "where", "why", "how", "?")):
            return "fact"
        if any(token in lowered for token in ("feel", "advice", "help me decide", "opinion")):
            return "coaching"
        return "general"











    @staticmethod
    def _normalize_key(text):
        return re.sub(r"\s+", " ", text.strip().lower())



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

    def _set_knowledge_context(self, intent, subject=""):
        self.conversation_state["last_knowledge_intent"] = intent
        self.conversation_state["last_knowledge_subject"] = subject



    @staticmethod
    def _pick_variant(options, seed_text):
        if not options:
            return ""
        seed = sum(ord(ch) for ch in str(seed_text))
        return options[seed % len(options)]








    @staticmethod
    def _parse_iso_date(value):
        value = value.strip()
        try:
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _math_expression_from_text(text):
        lower = text.lower().strip()
        patterns = [
            r"^(?:what is|what's|calculate|compute|solve)\s+(.+)$",
            r"^(.+)\s*=\s*\?$",
            r"^(.+)\s*=\s*$",
        ]

        expression = ""
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                expression = match.group(1).strip()
                break

        if not expression:
            return ""

        expression = expression.strip(" ?")

        replacements = {
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "x": "*",
            "divided by": "/",
            "over": "/",
            "mod": "%",
            "modulo": "%",
            "power of": "**",
            "to the power of": "**",
        }
        for source, target in replacements.items():
            expression = re.sub(rf"\b{re.escape(source)}\b", target, expression)

        expression = expression.replace("^", "**")
        expression = expression.replace(",", "")
        expression = re.sub(r"\s+", "", expression)

        if not re.fullmatch(r"[0-9+\-*/().%*]+", expression):
            return ""
        return expression

    @staticmethod
    def _eval_math_expression(expression):
        allowed_binary = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.FloorDiv: operator.floordiv,
        }
        allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Invalid constant")
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
                return allowed_binary[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
                return allowed_unary[type(node.op)](evaluate(node.operand))
            raise ValueError("Unsupported expression")

        parsed = ast.parse(expression, mode="eval")
        value = evaluate(parsed)

        if isinstance(value, float):
            value = round(value, 8)
        if abs(float(value)) > 1e15:
            raise ValueError("Result too large")
        return value

    def _math_answer(self, text):
        expression = self._math_expression_from_text(text)
        if not expression:
            return None

        try:
            result = self._eval_math_expression(expression)
        except (ZeroDivisionError, ValueError, OverflowError, SyntaxError):
            return None

        return f"The answer is {result}."

    def _percentage_answer(self, text):
        lowered = text.strip().lower()

        percent_of_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", lowered)
        if percent_of_match:
            pct = float(percent_of_match.group(1))
            base = float(percent_of_match.group(2))
            result = (pct / 100.0) * base
            return f"{pct}% of {base} is {round(result, 6)}."

        increase_match = re.search(r"increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%", lowered)
        if increase_match:
            base = float(increase_match.group(1))
            pct = float(increase_match.group(2))
            result = base * (1.0 + pct / 100.0)
            return f"Increasing {base} by {pct}% gives {round(result, 6)}."

        decrease_match = re.search(r"decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)%", lowered)
        if decrease_match:
            base = float(decrease_match.group(1))
            pct = float(decrease_match.group(2))
            result = base * (1.0 - pct / 100.0)
            return f"Decreasing {base} by {pct}% gives {round(result, 6)}."

        return None

    def _unit_conversion_answer(self, text):
        lowered = text.strip().lower()
        match = re.search(
            r"(?:convert|what is|how much is)\s+(-?\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:to|in)\s+([a-z]+)",
            lowered,
        )
        if not match:
            return None

        value = float(match.group(1))
        source = match.group(2)
        target = match.group(3)

        aliases = {
            "km": "km",
            "kilometer": "km",
            "kilometers": "km",
            "mile": "mi",
            "miles": "mi",
            "mi": "mi",
            "m": "m",
            "meter": "m",
            "meters": "m",
            "ft": "ft",
            "foot": "ft",
            "feet": "ft",
            "cm": "cm",
            "inch": "in",
            "inches": "in",
            "in": "in",
            "kg": "kg",
            "kilogram": "kg",
            "kilograms": "kg",
            "lb": "lb",
            "lbs": "lb",
            "pound": "lb",
            "pounds": "lb",
            "l": "l",
            "liter": "l",
            "liters": "l",
            "gal": "gal",
            "gallon": "gal",
            "gallons": "gal",
            "c": "c",
            "celsius": "c",
            "f": "f",
            "fahrenheit": "f",
        }

        src = aliases.get(source)
        dst = aliases.get(target)
        if not src or not dst:
            return None

        temp_pairs = {
            ("c", "f"): lambda x: (x * 9.0 / 5.0) + 32.0,
            ("f", "c"): lambda x: (x - 32.0) * 5.0 / 9.0,
        }
        if (src, dst) in temp_pairs:
            out = temp_pairs[(src, dst)](value)
            return f"{value} {src} is {round(out, 6)} {dst}."

        conversions = {
            ("km", "mi"): 0.621371,
            ("mi", "km"): 1.60934,
            ("m", "ft"): 3.28084,
            ("ft", "m"): 0.3048,
            ("cm", "in"): 0.393701,
            ("in", "cm"): 2.54,
            ("kg", "lb"): 2.20462,
            ("lb", "kg"): 0.453592,
            ("l", "gal"): 0.264172,
            ("gal", "l"): 3.78541,
        }
        factor = conversions.get((src, dst))
        if factor is None:
            if src == dst:
                return f"{value} {src} is {value} {dst}."
            return None

        out = value * factor
        return f"{value} {src} is {round(out, 6)} {dst}."

    def _date_reasoning_answer(self, text):
        lowered = text.strip().lower()

        day_match = re.search(r"what day is\s+(\d{4}-\d{2}-\d{2})", lowered)
        if day_match:
            parsed = self._parse_iso_date(day_match.group(1))
            if parsed:
                return f"{parsed.isoformat()} is a {parsed.strftime('%A')}."

        until_match = re.search(r"how many days until\s+(\d{4}-\d{2}-\d{2})", lowered)
        if until_match:
            parsed = self._parse_iso_date(until_match.group(1))
            if parsed:
                today = datetime.date.today()
                delta = (parsed - today).days
                if delta >= 0:
                    return f"There are {delta} days until {parsed.isoformat()}."
                return f"{parsed.isoformat()} was {-delta} days ago."

        between_match = re.search(
            r"how many days between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})",
            lowered,
        )
        if between_match:
            first = self._parse_iso_date(between_match.group(1))
            second = self._parse_iso_date(between_match.group(2))
            if first and second:
                return f"There are {abs((second - first).days)} days between those dates."

        return None

    def _offline_fact_answer(self, text):
        lowered = text.strip().lower().strip(" ?")

        capitals = {**OFFLINE_CAPITALS, **self.inference_manager.offline_knowledge.get("capitals", {})}
        people = {**OFFLINE_PEOPLE, **self.inference_manager.offline_knowledge.get("people", {})}
        local_facts = self.inference_manager.offline_knowledge.get("facts", {})
        facts = {**OFFLINE_FACTS, **local_facts}

        capital_match = re.search(r"capital of\s+([a-z\s]+)$", lowered)
        if capital_match:
            country = capital_match.group(1).strip()
            capital = capitals.get(country)
            if capital:
                return {
                    "answer": f"The capital of {country.title()} is {capital}.",
                    "intent": "capital",
                    "subject": country,
                }

        person_match = re.search(r"(?:who is|tell me about)\s+([a-z\s.]+)$", lowered)
        if person_match:
            name = person_match.group(1).strip().replace(".", "")
            answer = people.get(name)
            if answer:
                return {"answer": answer, "intent": "person", "subject": name}

        normalized = lowered.replace("?", "").strip()
        for key, answer in facts.items():
            if key == normalized or key in normalized:
                subject = normalized.replace("what is ", "").strip()
                if key in local_facts and subject and not answer.lower().startswith(subject.lower()):
                    answer = f"{subject.title()} is {answer}"
                if answer and answer[-1] not in ".!?":
                    answer = answer + "."
                return {"answer": answer, "intent": "definition", "subject": subject}

        fuzzy_answer = self.inference_manager._fuzzy_fact_answer(normalized)
        if fuzzy_answer:
            if fuzzy_answer and fuzzy_answer[-1] not in ".!?":
                fuzzy_answer = fuzzy_answer + "."
            return {
                "answer": fuzzy_answer,
                "intent": "definition",
                "subject": normalized.replace("what is ", "").strip(),
            }

        return None

    def _decision_coach_answer(self, text):
        lowered = text.strip().lower()

        should_match = re.search(r"^should i\s+(.+)$", lowered)
        if should_match:
            situation = should_match.group(1).strip(" ?")
            return (
                f"For '{situation}', use a quick decision rule: define your goal, list top two options, "
                "estimate impact and effort for each, then choose the option with higher long-term impact."
            )

        how_match = re.search(r"^how can i\s+(.+)$", lowered)
        if how_match:
            goal = how_match.group(1).strip(" ?")
            return (
                f"To {goal}, start with this structure: 1) define a clear outcome, 2) break it into three steps, "
                "3) execute the first step in a focused 20-minute block, 4) review and adjust."
            )

        compare_match = re.search(r"^which is better\s+(.+)\s+or\s+(.+)$", lowered)
        if compare_match:
            first = compare_match.group(1).strip(" ?")
            second = compare_match.group(2).strip(" ?")
            return (
                f"It depends on your priority. Compare {first} and {second} by speed, cost, and long-term benefit, "
                "then pick the one that matches your current constraint."
            )

        return None

    def _knowledge_answer(self, text):
        followup_answer = self.inference_manager._followup_knowledge_answer(text)
        if followup_answer:
            return followup_answer

        if not self._is_question(text):
            return None

        cache_key = text.strip().lower()
        if cache_key in self.knowledge_cache:
            return self.knowledge_cache[cache_key]

        answer = self._math_answer(text)
        if not answer:
            answer = self._percentage_answer(text)
        if not answer:
            answer = self._unit_conversion_answer(text)
        if not answer:
            answer = self._date_reasoning_answer(text)

        fact_result = None
        if not answer:
            fact_result = self._offline_fact_answer(text)
            if fact_result:
                answer = fact_result["answer"]
        if not answer:
            answer = self._decision_coach_answer(text)

        if answer:
            self.knowledge_cache[cache_key] = answer
            if len(self.knowledge_cache) > 120:
                # Keep memory bounded while preserving recent lookups.
                oldest_key = next(iter(self.knowledge_cache))
                self.knowledge_cache.pop(oldest_key, None)

            if fact_result:
                self._set_knowledge_context(fact_result.get("intent", ""), fact_result.get("subject", ""))
            else:
                self._set_knowledge_context("", "")

            return answer

        # Allow outer response pipeline to try web lookup for unanswered factual questions.
        return None

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

    def _load_daily_planner(self):
        default_data = {"tasks": [], "reminders": []}
        if not TASKS_PATH.exists():
            return default_data

        try:
            with TASKS_PATH.open("r", encoding="utf-8") as file_handle:
                loaded = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return default_data

        if not isinstance(loaded, dict):
            return default_data

        tasks = loaded.get("tasks", [])
        reminders = loaded.get("reminders", [])

        if not isinstance(tasks, list):
            tasks = []
        if not isinstance(reminders, list):
            reminders = []

        cleaned_tasks = []
        for item in tasks:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            cleaned_tasks.append(
                {
                    "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
                    "text": text,
                    "done": bool(item.get("done", False)),
                    "created_at": str(item.get("created_at", "")).strip() or self._timestamp_now(),
                    "due": str(item.get("due", "")).strip(),
                }
            )

        cleaned_reminders = []
        for item in reminders:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            cleaned_reminders.append(
                {
                    "id": int(item.get("id", 0)) if str(item.get("id", "")).isdigit() else 0,
                    "text": text,
                    "when": str(item.get("when", "")).strip(),
                    "done": bool(item.get("done", False)),
                    "created_at": str(item.get("created_at", "")).strip() or self._timestamp_now(),
                }
            )

        return {"tasks": cleaned_tasks[-200:], "reminders": cleaned_reminders[-200:]}

    def _save_daily_planner(self):
        TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TASKS_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.daily_planner, file_handle, indent=2)

    @staticmethod
    def _next_item_id(items):
        highest = 0
        for item in items:
            try:
                highest = max(highest, int(item.get("id", 0)))
            except (TypeError, ValueError):
                continue
        return highest + 1

    @staticmethod
    def _extract_first_number(text):
        match = re.search(r"\b(\d+)\b", text)
        if not match:
            return None
        return int(match.group(1))

    def _operator_label(self):
        name = self.operator_memory.get("operator_name", "").strip()
        return name if name else "there"

    def _remember_personal_fact(self, fact):
        fact = fact.strip(" .")
        if not fact:
            return

        facts = self.operator_memory.setdefault("personal_facts", [])
        if fact not in facts:
            facts.append(fact)
        self.operator_memory["personal_facts"] = facts[-25:]
        self._save_operator_memory()

    def _personal_facts_summary(self):
        facts = self.operator_memory.get("personal_facts", [])
        if not facts:
            return "I know your preferred name and current notes, but not many personal details yet."

        joined = "; ".join(facts[-3:])
        return f"From our conversations: {joined}."

    def _infer_topic(self, text):
        if any(token in text for token in ("task", "todo", "priority", "plan")):
            return "planning"
        if "remind" in text or "reminder" in text:
            return "reminders"
        if any(token in text for token in ("feel", "stressed", "tired", "sad", "happy", "anxious")):
            return "wellbeing"
        if any(token in text for token in ("name", "who am i", "about me", "remember")):
            return "identity"
        if any(token in text for token in ("work", "study", "career", "project")):
            return "work"
        if any(token in text for token in ("hello", "hi", "thanks", "how are you")):
            return "smalltalk"
        return "general"

    def _update_conversation_state(self, user_text, assistant_text):
        if not user_text:
            return

        self._capture_preference(user_text)

        self.conversation_state["last_topic"] = self._infer_topic(user_text)
        self.conversation_state["last_user_mood"] = self.inference_manager._detect_mood(user_text)

        goal_phrase = self.inference_manager._extract_goal_phrase(user_text)
        if goal_phrase:
            self.conversation_state["last_goal"] = goal_phrase

        self.conversation_state["last_user_text"] = user_text
        self.conversation_state["last_assistant_text"] = assistant_text or ""
        self.conversation_state["turn_count"] = int(self.conversation_state.get("turn_count", 0)) + 1
        self._record_interaction_for_training(user_text, assistant_text)

    def _followup_response(self, text):
        followup_triggers = {
            "tell me more",
            "go on",
            "continue",
            "why",
            "how so",
            "what do you mean",
            "explain",
        }
        if text not in followup_triggers:
            return None

        topic = self.conversation_state.get("last_topic", "")
        if topic == "planning":
            return (
                "For planning, pick one high-impact task, two medium tasks, and one quick win. "
                "I can help you convert that into a concrete list right now."
            )
        if topic == "wellbeing":
            return (
                "If stress is high, a short reset helps: hydrate, take a two-minute breathing break, "
                "then start with the smallest useful action."
            )
        if topic == "work":
            return (
                "For work progress, define the next visible deliverable and a 25-minute focus block. "
                "I can set tasks for both."
            )
        if topic:
            return "I can go deeper on that. Tell me the exact part you want to focus on."
        return "Certainly. Tell me the topic and I will break it down step by step."

    def _human_chat_response(self, text):
        if "what do you know about me" in text or "tell me about me" in text:
            notes_count = len(self.operator_memory.get("notes", []))
            return f"{self._personal_facts_summary()} I also have {notes_count} saved notes."

        if "what were we talking about" in text or "last topic" in text:
            topic = self.conversation_state.get("last_topic", "")
            if topic:
                return f"Our last topic was {topic}."
            return "We have not established a clear topic yet."

        followup = self._followup_response(text)
        if followup:
            return followup

        if "let's chat" in text or "lets chat" in text or "talk to me" in text:
            return "I am here with you. We can talk about your day, your goals, or anything on your mind."

        opinion_match = re.search(r"(?:what do you think about|your thoughts on)\s+(.+)$", text)
        if opinion_match:
            subject = opinion_match.group(1).strip(" .")
            return (
                f"My take on {subject}: start from your goal, compare benefits, risks, and effort, "
                "then choose the option that helps your long-term direction."
            )

        advice_match = re.search(r"(?:i need advice on|help me decide on|help me with)\s+(.+)$", text)
        if advice_match:
            subject = advice_match.group(1).strip(" .")
            return (
                f"Sure. For {subject}, we can do this quickly: define success, list your options, "
                "score each option from 1 to 10, and pick the strongest one."
            )

        personal_patterns = [
            r"\bi like\s+(.+)",
            r"\bi love\s+(.+)",
            r"\bmy favorite\s+(.+)",
            r"\bi work as\s+(.+)",
            r"\bi study\s+(.+)",
            r"\bi live in\s+(.+)",
        ]
        for pattern in personal_patterns:
            match = re.search(pattern, text)
            if match:
                fact = match.group(0).strip(" .")
                self._remember_personal_fact(fact)
                return "Noted. I will remember that for future conversations."

        if "i feel" in text or re.search(r"\bi am\s+(sad|stressed|tired|upset|anxious|overwhelmed)\b", text):
            return (
                f"I hear you, {self._operator_label()}. Want a quick reset plan: "
                "one small task, one break, then a focused 20-minute sprint?"
            )

        if re.search(r"\bi am\s+(good|fine|happy|great|excited|motivated)\b", text):
            return "Great energy. We can use it well, want me to line up your top priorities now?"

        if "i am bored" in text or "i'm bored" in text:
            return (
                "I can give you a quick productivity challenge: finish one small task in 10 minutes, "
                "then we pick the next one."
            )

        if "can we talk" in text:
            return "Absolutely. Tell me what is on your mind, and I will stay focused with you."

        if "what should i do next" in text or "what now" in text:
            last_goal = self.conversation_state.get("last_goal", "")
            if last_goal:
                return f"Next step for {last_goal}: start a focused 20-minute block on the first actionable task."
            return "Pick one meaningful task you can finish in 20 minutes, start it now, and we will iterate from there."

        return None

    def _task_overview(self, include_completed=False, limit=6):
        tasks = self.daily_planner.get("tasks", [])
        if not include_completed:
            tasks = [task for task in tasks if not task.get("done")]

        if not tasks:
            return "No active tasks. You are clear for now."

        lines = []
        for task in tasks[:limit]:
            due = task.get("due", "")
            suffix = f" (due {due})" if due else ""
            lines.append(f"task {task.get('id')}: {task.get('text')}{suffix}")

        if len(tasks) > limit:
            lines.append(f"plus {len(tasks) - limit} more")

        return "; ".join(lines) + "."

    def _reminder_overview(self, include_completed=False, limit=5):
        reminders = self.daily_planner.get("reminders", [])
        if not include_completed:
            reminders = [item for item in reminders if not item.get("done")]

        if not reminders:
            return "No active reminders."

        lines = []
        for reminder in reminders[:limit]:
            when = reminder.get("when", "")
            suffix = f" at {when}" if when else ""
            lines.append(f"reminder {reminder.get('id')}: {reminder.get('text')}{suffix}")

        if len(reminders) > limit:
            lines.append(f"plus {len(reminders) - limit} more")

        return "; ".join(lines) + "."

    def _add_task(self, text, due=""):
        tasks = self.daily_planner.setdefault("tasks", [])
        task_id = self._next_item_id(tasks)
        tasks.append(
            {
                "id": task_id,
                "text": text,
                "done": False,
                "created_at": self._timestamp_now(),
                "due": due,
            }
        )
        self.daily_planner["tasks"] = tasks[-200:]
        self._save_daily_planner()

        if due:
            return f"Task {task_id} added: {text}, due {due}."
        return f"Task {task_id} added: {text}."

    def _set_task_done(self, task_id):
        for task in self.daily_planner.get("tasks", []):
            if int(task.get("id", 0)) == task_id:
                if task.get("done"):
                    return f"Task {task_id} is already completed."
                task["done"] = True
                self._save_daily_planner()
                return f"Task {task_id} completed."
        return f"I could not find task {task_id}."

    def _remove_task(self, task_id):
        tasks = self.daily_planner.get("tasks", [])
        kept = [task for task in tasks if int(task.get("id", 0)) != task_id]
        if len(kept) == len(tasks):
            return f"I could not find task {task_id}."
        self.daily_planner["tasks"] = kept
        self._save_daily_planner()
        return f"Task {task_id} removed."

    def _clear_completed_tasks(self):
        tasks = self.daily_planner.get("tasks", [])
        active = [task for task in tasks if not task.get("done")]
        removed = len(tasks) - len(active)
        self.daily_planner["tasks"] = active
        self._save_daily_planner()
        return f"Cleared {removed} completed tasks."

    def _add_reminder(self, text, when=""):
        reminders = self.daily_planner.setdefault("reminders", [])
        reminder_id = self._next_item_id(reminders)
        reminders.append(
            {
                "id": reminder_id,
                "text": text,
                "when": when,
                "done": False,
                "created_at": self._timestamp_now(),
            }
        )
        self.daily_planner["reminders"] = reminders[-200:]
        self._save_daily_planner()

        if when:
            return f"Reminder {reminder_id} added for {when}: {text}."
        return f"Reminder {reminder_id} added: {text}."

    def _set_reminder_done(self, reminder_id):
        for reminder in self.daily_planner.get("reminders", []):
            if int(reminder.get("id", 0)) == reminder_id:
                if reminder.get("done"):
                    return f"Reminder {reminder_id} is already completed."
                reminder["done"] = True
                self._save_daily_planner()
                return f"Reminder {reminder_id} completed."
        return f"I could not find reminder {reminder_id}."

    def _daily_brief(self):
        now = datetime.datetime.now()
        today = now.strftime("%A, %B %d")
        current_time = now.strftime("%I:%M %p")

        pending_tasks = [task for task in self.daily_planner.get("tasks", []) if not task.get("done")]
        active_reminders = [
            reminder for reminder in self.daily_planner.get("reminders", []) if not reminder.get("done")
        ]

        if pending_tasks:
            top_task = pending_tasks[0].get("text", "")
            tasks_line = f"{len(pending_tasks)} active tasks. Next priority: {top_task}."
        else:
            tasks_line = "No active tasks. You can add one by saying add task followed by the task."

        reminders_line = (
            f"{len(active_reminders)} active reminders."
            if active_reminders
            else "No active reminders."
        )

        return f"Daily brief for {today}, {current_time}. {tasks_line} {reminders_line}"

    @staticmethod
    def _format_duration(seconds):
        seconds = int(max(0, seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        parts = []
        if hours:
            parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
        if minutes:
            parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
        if secs or not parts:
            parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
        return " ".join(parts)

    @staticmethod
    def _parse_timer_duration_seconds(text):
        units = re.findall(
            r"(\d+)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)",
            text.lower(),
        )
        if not units:
            return None

        total = 0
        for amount_raw, unit in units:
            amount = int(amount_raw)
            if amount < 0:
                return None
            if unit.startswith(("hour", "hr", "h")):
                total += amount * 3600
            elif unit.startswith(("minute", "min", "m")):
                total += amount * 60
            else:
                total += amount

        if total <= 0 or total > 24 * 3600:
            return None
        return total

    @staticmethod
    def _parse_clock_time(value):
        probe = value.strip().lower().replace(".", "")
        for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
            try:
                parsed = datetime.datetime.strptime(probe.upper(), fmt)
                return parsed.hour, parsed.minute
            except ValueError:
                continue
        return None

    def _set_timer(self, seconds, label="timer"):
        timer_id = len(self.active_timers) + 1
        self.active_timers.append(
            {
                "id": timer_id,
                "label": label,
                "due": time.monotonic() + float(seconds),
                "duration": int(seconds),
            }
        )
        return f"Timer {timer_id} set for {self._format_duration(seconds)}."

    def _list_timers(self):
        if not self.active_timers:
            return "No active timers."

        now = time.monotonic()
        parts = []
        for timer in self.active_timers[:5]:
            remaining = max(0, int(timer["due"] - now))
            parts.append(
                f"timer {timer['id']} {timer['label']} with {self._format_duration(remaining)} remaining"
            )
        if len(self.active_timers) > 5:
            parts.append(f"plus {len(self.active_timers) - 5} more")
        return "; ".join(parts) + "."

    def _cancel_timer(self, timer_id=None):
        if not self.active_timers:
            return "No active timers to cancel."

        if timer_id is None:
            self.active_timers.clear()
            return "All active timers canceled."

        kept = [timer for timer in self.active_timers if int(timer.get("id", 0)) != timer_id]
        if len(kept) == len(self.active_timers):
            return f"I could not find timer {timer_id}."

        self.active_timers = kept
        return f"Timer {timer_id} canceled."

    def _check_due_timers(self):
        if not self.active_timers:
            return []

        now = time.monotonic()
        due = [timer for timer in self.active_timers if timer["due"] <= now]
        if not due:
            return []

        due_ids = {timer["id"] for timer in due}
        self.active_timers = [timer for timer in self.active_timers if timer["id"] not in due_ids]
        return [
            f"Timer {timer['id']} complete for {timer.get('label', 'timer')}."
            for timer in due
        ]

    def _check_due_reminders(self):
        now = datetime.datetime.now()
        tick_key = now.strftime("%Y-%m-%d %H:%M")
        if tick_key == self._last_reminder_tick:
            return []
        self._last_reminder_tick = tick_key

        fired = []
        for reminder in self.daily_planner.get("reminders", []):
            if reminder.get("done"):
                continue
            when_text = str(reminder.get("when", "")).strip()
            if not when_text:
                continue

            parsed = self._parse_clock_time(when_text)
            if not parsed:
                continue

            hour, minute = parsed
            if now.hour == hour and now.minute == minute:
                reminder["done"] = True
                fired.append(f"Reminder {reminder.get('id')}: {reminder.get('text')}")

        if fired:
            self._save_daily_planner()
        return fired

    def _is_duplicate_utterance(self, text):
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False

        now = time.monotonic()
        if (
            normalized == self._last_handled_text
            and now - self._last_handled_at <= DUPLICATE_UTTERANCE_WINDOW_SECONDS
        ):
            return True

        self._last_handled_text = normalized
        self._last_handled_at = now
        return False

    def _bulk_add_tasks(self, task_texts):
        task_texts = [text.strip() for text in task_texts if text and text.strip()]
        if not task_texts:
            return []

        tasks = self.daily_planner.setdefault("tasks", [])
        next_id = self._next_item_id(tasks)
        added_ids = []
        for text in task_texts:
            tasks.append(
                {
                    "id": next_id,
                    "text": text,
                    "done": False,
                    "created_at": self._timestamp_now(),
                    "due": "",
                }
            )
            added_ids.append(next_id)
            next_id += 1

        self.daily_planner["tasks"] = tasks[-200:]
        self._save_daily_planner()
        return added_ids

    def _generate_goal_steps(self, goal):
        goal_lower = goal.lower()
        if any(token in goal_lower for token in ("study", "learn", "exam", "course")):
            return [
                f"Define the exact learning target for {goal}",
                "Split the topic into four focused subtopics",
                "Create a daily 45-minute study block with no distractions",
                "Practice with exercises and track weak areas",
                "Review progress at end of day and adjust next session",
            ]
        if any(token in goal_lower for token in ("project", "build", "app", "system")):
            return [
                f"Write a one-sentence objective for {goal}",
                "Break implementation into milestone tasks",
                "Deliver a minimal working version first",
                "Test core flows and fix blockers",
                "Polish, document, and finalize",
            ]
        return [
            f"Clarify success criteria for {goal}",
            "Break the goal into five executable actions",
            "Schedule the first action in your next focus block",
            "Track completion and remove blockers daily",
            "Review results and set next iteration",
        ]

    def _build_goal_plan(self, goal, auto_add_tasks=False):
        steps = self._generate_goal_steps(goal)
        self.conversation_state["last_plan_goal"] = goal
        self.conversation_state["last_plan_steps"] = steps

        if auto_add_tasks:
            added_ids = self._bulk_add_tasks(steps)
            listed = ", ".join(str(i) for i in added_ids)
            return (
                f"Plan created for {goal}. I added {len(added_ids)} tasks to your list "
                f"with IDs {listed}."
            )

        numbered = " ".join(f"step {index + 1}: {step}." for index, step in enumerate(steps))
        return f"Strategic plan for {goal}: {numbered}"


    def _activate_session(self):
        self.active_until = time.monotonic() + ACTIVE_WINDOW_SECONDS

    def _session_active(self):
        return time.monotonic() < self.active_until

    def set_conversational_mode(self, enabled):
        self.conversational_mode = bool(enabled)

    @staticmethod
    def _contains_phrase(text, phrase):
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None

    def _configure_voice(self):
        self.engine.setProperty("rate", 182)
        self.engine.setProperty("volume", 1.0)

        voices = self.engine.getProperty("voices")
        for voice in voices:
            voice_name = (voice.name or "").lower()
            if "english" in voice_name and ("zira" in voice_name or "david" in voice_name):
                self.engine.setProperty("voice", voice.id)
                break

    @staticmethod
    def _console_print(text):
        value = str(text)
        try:
            print(value)
            return
        except UnicodeEncodeError:
            pass

        encoding = (sys.stdout.encoding or "utf-8")
        safe = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)

    def speak(self, text):
        self._console_print(f"Friday: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    @staticmethod
    def _spoken_token_to_letter(token):
        # Convert common letter pronunciations produced by STT to plain letters.
        mapping = {
            "a": "a",
            "ay": "a",
            "b": "b",
            "be": "b",
            "bee": "b",
            "c": "c",
            "see": "c",
            "sea": "c",
            "d": "d",
            "dee": "d",
            "e": "e",
            "f": "f",
            "ef": "f",
            "g": "g",
            "gee": "g",
            "h": "h",
            "i": "i",
            "eye": "i",
            "j": "j",
            "jay": "j",
            "k": "k",
            "kay": "k",
            "l": "l",
            "el": "l",
            "m": "m",
            "em": "m",
            "n": "n",
            "en": "n",
            "o": "o",
            "oh": "o",
            "p": "p",
            "pea": "p",
            "q": "q",
            "queue": "q",
            "r": "r",
            "are": "r",
            "s": "s",
            "ess": "s",
            "t": "t",
            "tea": "t",
            "u": "u",
            "v": "v",
            "vee": "v",
            "vv": "bb",
            "w": "w",
            "doubleu": "w",
            "x": "x",
            "ex": "x",
            "y": "y",
            "z": "z",
            "zee": "z",
            "zed": "z",
        }
        return mapping.get(token, "")

    def _extract_directive(self, raw_text):
        text = " ".join(str(raw_text or "").strip().lower().split())
        if not text:
            return "", False

        wake_used = False
        for wake in WAKE_WORDS:
            if text == wake:
                return "", True
            if text.startswith(wake + " "):
                return text[len(wake):].strip(" ,"), True

        for alias in WAKE_ALIASES:
            if text == alias:
                return "", True
            if text.startswith(alias + " "):
                return text[len(alias):].strip(" ,"), True

        first_token = text.split(" ", 1)[0]
        if first_token and difflib.SequenceMatcher(None, first_token, "friday").ratio() >= 0.76:
            remainder = text[len(first_token):].strip(" ,")
            return remainder, True

        if " friday " in f" {text} ":
            wake_used = True
            text = re.sub(r"\bhey\s+friday\b|\bfriday\b", "", text).strip(" ,")

        if not wake_used:
            for alias in WAKE_ALIASES:
                if f" {alias} " in f" {text} ":
                    wake_used = True
                    text = re.sub(rf"\b{re.escape(alias)}\b", "", text).strip(" ,")
                    break

        return text, wake_used

    def _intent_response(self, text, wake_used):
        if any(self._contains_phrase(text, word) for word in EXIT_WORDS):
            return "__EXIT__"

        if (
            text.strip() == "conversational mode"
            or "conversational mode on" in text
            or "enable conversational mode" in text
        ):
            self.conversational_mode = True
            return "Conversational mode enabled. You can speak naturally now."

        if "conversational mode off" in text or "disable conversational mode" in text:
            self.conversational_mode = False
            return "Conversational mode disabled. Use my wake word again."

        if "go to sleep" in text or "sleep mode" in text:
            self.sleep_mode = True
            self.active_until = 0.0
            return "Entering standby mode. Say Friday to wake me."

        if self.sleep_mode:
            if wake_used or "wake up" in text:
                self.sleep_mode = False
                self._activate_session()
                return "Online and listening."
            return None

        if text in {"", "listen", "are you there"} and wake_used:
            return "Listening."

        if "how are you" in text:
            return "I am doing well and fully focused. How are you feeling today?"

        if re.search(r"\b(hello|hi|hey)\b", text):
            return "Hey, good to hear from you. What would you like to work on right now?"

        if "thank you" in text or "thanks" in text:
            return "Always. I am here whenever you need me."

        if "good morning" in text:
            return f"Good morning, {self._operator_label()}. Ready for your daily brief when you are."

        if "good evening" in text or "good night" in text:
            return "Good evening. I can help you wrap up tasks and set tomorrow's priorities."

        # Explicit quick answers for date and year to avoid generic fallback responses
        if re.search(r"\b(?:what(?:'s| is)?\s+the\s+date|what\s+date|date\s+is\s+today|today\s+is)\b", text):
            return datetime.datetime.now().strftime("%A, %B %d, %Y")

        if re.search(r"\b(?:what(?:'s| is)?\s+the\s+year|what\s+year|current\s+year|year\s+is\s+it)\b", text):
            return f"The current year is {datetime.datetime.now().year}."

        # Time queries
        if re.search(r"\b(?:what(?:'s| is)?\s+the\s+time|what\s+time|current\s+time|time\s+is\s+it)\b", text):
            return datetime.datetime.now().strftime("%H:%M:%S")

        tz_match = re.search(r"what time is it in ([a-zA-Z/ _-]+)", text)
        if tz_match:
            try:
                import pytz
                tz_name = tz_match.group(1).strip().replace(' ', '_')
                tz = pytz.timezone(tz_name)
                return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                return "I couldn't resolve that timezone. Try a TZ like Europe/London or America/New_York."

        # Sunrise / sunset handlers
        sun_match = re.search(r"\b(?:sunrise|sunset)\b(?: in ([a-zA-Z ,/\-_]+))?", text)
        if sun_match:
            place = sun_match.group(1)
            try:
                from astral import LocationInfo
                from astral.sun import sun
                if place:
                    name = place.split(',')[0].strip()
                    loc = LocationInfo(name)
                else:
                    loc = LocationInfo("Local", "", "UTC", 0, 0)
                s = sun(loc.observer, date=datetime.date.today())
                when = s.get("sunrise") if "sunrise" in text else s.get("sunset")
                if when:
                    return when.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                return "I couldn't compute sunrise/sunset for that location. Try a city name like London or New_York."

        learned_now = self.inference_manager._teach_local_knowledge(text)
        if learned_now:
            return learned_now

        learned_nlp = self._teach_nlp_pair(text)
        if learned_nlp:
            return learned_nlp

        if (
            "retrain model" in text
            or "train model" in text
            or "retrain deep model" in text
            or "train deep model" in text
        ):
            subprocess.Popen([sys.executable, "train.py"])
            return (
                "Started deep-learning retraining in the background. "
                "When it finishes, say reload generator to use new weights."
            )

        if "reload generator" in text or "refresh generator" in text:
            ready = reload_generator()
            if ready:
                return "Generator reloaded with current model weights."
            return "Generator reloaded in safe mode. Train the model first for richer output."

        operator_name_match = re.search(r"\b(?:call me|my name is)\s+([a-z][a-z\- ]{1,30})$", text)
        if operator_name_match:
            operator_name = operator_name_match.group(1).strip().title()
            self.operator_memory["operator_name"] = operator_name
            self._save_operator_memory()
            return f"Acknowledged. I will call you {operator_name}."

        if "who am i" in text or "what is my name" in text:
            operator_name = self.operator_memory.get("operator_name")
            if operator_name:
                return f"You are {operator_name}."
            return "You have not set a preferred name yet. Say call me followed by your name."

        remember_match = re.search(r"\bremember that\s+(.+)", text)
        if remember_match:
            note = remember_match.group(1).strip(" .")
            if note:
                notes = self.operator_memory.setdefault("notes", [])
                notes.append(note)
                self.operator_memory["notes"] = notes[-20:]
                self._save_operator_memory()
                return "Stored in memory bank."
            return "Please tell me what to remember."

        if "what do you remember" in text or "memory report" in text:
            notes = self.operator_memory.get("notes", [])
            if not notes:
                return "No saved notes in memory bank."
            if len(notes) == 1:
                return f"One saved note: {notes[0]}."
            joined = "; ".join(notes[-3:])
            return f"I have {len(notes)} saved notes. Latest entries: {joined}."

        if "forget memory" in text or "clear memory" in text:
            self.operator_memory["notes"] = []
            self._save_operator_memory()
            return "Memory bank cleared."

        if "daily brief" in text or "plan my day" in text or "what should i do today" in text:
            return self._daily_brief()

        auto_plan_match = re.search(
            r"\b(?:auto plan|plan and add tasks for|build plan and tasks for)\s+(.+)",
            text,
        )
        if auto_plan_match:
            goal = auto_plan_match.group(1).strip(" .")
            if not goal:
                return "Tell me the goal you want planned."
            return self._build_goal_plan(goal, auto_add_tasks=True)

        plan_match = re.search(r"\b(?:plan goal|make a plan for|roadmap for|plan for)\s+(.+)", text)
        if plan_match:
            goal = plan_match.group(1).strip(" .")
            if not goal:
                return "Tell me the goal you want planned."
            return self._build_goal_plan(goal, auto_add_tasks=False)

        if "add that plan to tasks" in text or "turn that plan into tasks" in text:
            steps = self.conversation_state.get("last_plan_steps", [])
            goal = self.conversation_state.get("last_plan_goal", "")
            if not steps:
                return "No recent plan found. Ask me to plan a goal first."
            added_ids = self._bulk_add_tasks(steps)
            listed = ", ".join(str(i) for i in added_ids)
            return f"Added the latest plan for {goal} into tasks with IDs {listed}."

        timer_set_match = re.search(r"\b(?:set|start)\s+(?:a\s+)?timer(?:\s+for)?\s+(.+)$", text)
        if timer_set_match:
            timer_payload = timer_set_match.group(1).strip(" .")
            seconds = self._parse_timer_duration_seconds(timer_payload)
            if seconds is None:
                return "Tell me timer duration like set timer for 10 minutes or 45 seconds."

            label_match = re.search(
                r"\d+\s*(?:hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\s+(?:for|to)\s+(.+)$",
                timer_payload,
            )
            label = (label_match.group(1).strip(" .") if label_match else "timer")
            return self._set_timer(seconds, label=label)

        if "list timers" in text or "show timers" in text or "my timers" in text:
            return self._list_timers()

        if "cancel all timers" in text or "clear timers" in text:
            return self._cancel_timer()

        if "cancel timer" in text or "stop timer" in text:
            timer_id = self._extract_first_number(text)
            return self._cancel_timer(timer_id)

        add_task_match = re.search(
            r"\b(?:add|create)\s+(?:a\s+)?(?:task|todo)\s+(?:to\s+)?(.+)",
            text,
        )
        if add_task_match:
            task_text = add_task_match.group(1).strip(" .")
            due_match = re.search(r"\bby\s+(.+)$", task_text)
            due = ""
            if due_match:
                due = due_match.group(1).strip(" .")
                task_text = task_text[: due_match.start()].strip(" .")

            if not task_text:
                return "Tell me the task details, for example: add task send report by 5 pm."
            return self._add_task(task_text, due)

        if "list tasks" in text or "show tasks" in text or "my tasks" in text or "todo list" in text:
            return self._task_overview(include_completed=False)

        if "list all tasks" in text or "show completed tasks" in text:
            return self._task_overview(include_completed=True)

        if "clear completed tasks" in text:
            return self._clear_completed_tasks()

        if "clear all tasks" in text:
            self.daily_planner["tasks"] = []
            self._save_daily_planner()
            return "All tasks cleared."

        if re.search(r"\b(?:complete|finish|mark)\b", text) and "task" in text:
            task_id = self._extract_first_number(text)
            if task_id is None:
                return "Tell me which task number to complete, for example complete task 2."
            return self._set_task_done(task_id)

        if re.search(r"\b(?:remove|delete|cancel)\b", text) and "task" in text:
            task_id = self._extract_first_number(text)
            if task_id is None:
                return "Tell me which task number to remove, for example remove task 2."
            return self._remove_task(task_id)

        reminder_match = re.search(r"\bremind me to\s+(.+?)(?:\s+at\s+(.+))?$", text)
        if reminder_match:
            reminder_text = reminder_match.group(1).strip(" .")
            when = (reminder_match.group(2) or "").strip(" .")
            if not reminder_text:
                return "Tell me what you want a reminder for."
            return self._add_reminder(reminder_text, when)

        if "list reminders" in text or "show reminders" in text or "my reminders" in text:
            return self._reminder_overview(include_completed=False)

        if "list all reminders" in text or "show completed reminders" in text:
            return self._reminder_overview(include_completed=True)

        if "clear all reminders" in text:
            self.daily_planner["reminders"] = []
            self._save_daily_planner()
            return "All reminders cleared."

        if re.search(r"\b(?:complete|finish|mark)\b", text) and "reminder" in text:
            reminder_id = self._extract_first_number(text)
            if reminder_id is None:
                return "Tell me which reminder number to complete."
            return self._set_reminder_done(reminder_id)

        if "status report" in text or "system status" in text:
            return "All systems are stable. Voice input and response modules are operational."

        if "mission brief" in text:
            now_time = datetime.datetime.now().strftime("%I:%M %p")
            today = datetime.datetime.now().strftime("%A, %B %d")
            notes_count = len(self.operator_memory.get("notes", []))
            pending_tasks = len([task for task in self.daily_planner.get("tasks", []) if not task.get("done")])
            return (
                f"Mission brief: local time {now_time}, date {today}, "
                f"{notes_count} items stored in memory bank, {pending_tasks} active tasks, "
                "and all assistant systems nominal."
            )

        if "time" in text:
            now_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now_time}."

        if "date" in text or text in {"what day is it", "which day is it", "what is today"}:
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {today}."

        for app_name, (command, response_text) in APP_OPEN_COMMANDS.items():
            if f"open {app_name}" in text or f"launch {app_name}" in text:
                subprocess.Popen(command)
                return response_text

        for app_name, image_name in APP_CLOSE_IMAGES.items():
            if f"close {app_name}" in text:
                result = subprocess.run(
                    ["taskkill", "/IM", image_name, "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return f"Closed {app_name}."
                return f"{app_name.title()} was not running."

        if "open cmd" in text:
            subprocess.Popen("cmd")
            return "Opening Command Prompt."

        if "open browser" in text:
            webbrowser.open("https://www.google.com")
            return "Opening your browser."

        if "open youtube" in text:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube."

        if "open github" in text:
            webbrowser.open("https://github.com")
            return "Opening GitHub."

        search_match = re.search(r"(?:search for|google)\s+(.+)", text)
        if search_match:
            query = search_match.group(1).strip()
            if query:
                webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
                return f"Searching the web for {query}."

        if "who are you" in text or "your name" in text:
            return "I am FRIDAY, your AI desktop assistant."

        if "show memory tiers" in text or "memory tiers" in text:
            short_count = len(self.session_memory.get("recent_turns", []))
            pref_count = len(self.user_preferences.get("preferences", []))
            task_count = len(self.daily_planner.get("tasks", []))
            return (
                f"Memory tiers report: short-term session memory has {short_count} turns, "
                f"long-term preferences has {pref_count} items, task memory has {task_count} tasks."
            )

        if "list preferences" in text or "show preferences" in text:
            prefs = self.user_preferences.get("preferences", [])
            if not prefs:
                return "No long-term preferences saved yet."
            return "Saved preferences: " + "; ".join(prefs[-8:]) + "."

        if "help" in text:
            return (
                "Use wake phrase Friday, then ask daily commands like add task, list tasks, "
                "complete task 2, remind me to call mom at 7 pm, daily brief, plus time, date, "
                "open or close apps, web search, memory commands, and offline smart Q&A like "
                "math, percentages, unit conversion, date calculations, capitals, and decision coaching. "
                "Advanced commands: set timer for 10 minutes, plan goal launch startup, auto plan learn python, "
                "teach facts with learn that gravity is a force, teach Q and A with "
                "learn qa what is gpu ||| a gpu is a parallel processor, retrain model, "
                "and reload generator."
            )

        knowledge = self._knowledge_answer(text)
        if knowledge:
            return knowledge

        if self._is_question(text):
            local_qa = self.inference_manager._dataset_qa_answer(text)
            if local_qa:
                return local_qa

            web_answer = self.inference_manager._web_answer(text)
            if web_answer:
                return web_answer

        conversational = self._human_chat_response(text)
        if conversational:
            return conversational

        semantic = self.inference_manager._semantic_qa_response(text)
        if semantic:
            return semantic

        return None

    def _fallback_response(self, text):
        self.history.append(text)
        self.history = self.history[-20:]

        if len(text.split()) <= 2:
            return "I am with you. Give me a little more detail, and I will help clearly."

        if self._is_question(text):
            planned = self.inference_manager._planner_response(text)
            if planned:
                return planned
            return (
                "I am not fully certain based on current grounded data. "
                "Please rephrase or provide the exact topic to ground the answer."
            )

        planned = self.inference_manager._planner_response(text)
        if planned:
            return planned

        return "I can help once I have grounded context. Tell me your exact goal or facts to use."

    def respond(self, raw_text):
        normalized, wake_used = self._extract_directive(raw_text)

        mode_command = normalized.strip() in {
            "conversational mode",
            "conversational mode on",
            "enable conversational mode",
            "conversational mode off",
            "disable conversational mode",
        }

        if wake_used:
            self._activate_session()

        if not wake_used and not self._session_active() and not self.conversational_mode and not mode_command:
            return None

        if not normalized and wake_used:
            return "Listening."

        response = self._intent_response(normalized, wake_used)
        if response == "__EXIT__":
            return response
        if response:
            self._update_conversation_state(normalized, response)
            return response

        fallback = self._fallback_response(normalized)
        self._update_conversation_state(normalized, fallback)
        return fallback

    def run(self):
        self.speak('System online Sir. Say ''quit'' to exit.')
        self.microphone = sr.Microphone()

        with self.microphone as source:
            print("Calibrating background noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
            self.recognizer.dynamic_energy_threshold = True

        while True:
            try:
                if hasattr(self, '_check_due_timers'):
                    alerts = self._check_due_timers()
                    alerts.extend(self._check_due_reminders())
                    for alert in alerts:
                        self.speak(alert)

                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)

                text = self.recognizer.recognize_google(audio).lower()
                print(f"You: {text}")

                if self._is_duplicate_utterance(text):
                    continue

                response = self.respond(text)
                if not response:
                    continue

                if response == '__EXIT__':
                    self.speak('Powering down Sir. Goodbye.')
                    break

                self.speak(response)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Google SR error: {e}")
            except Exception as e:
                print(f"Runtime error: {e}")
                
if __name__ == "__main__":
    assistant = FridayAssistant()
    assistant.run()
                