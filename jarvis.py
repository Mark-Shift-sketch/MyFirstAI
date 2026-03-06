"""Voice assistant loop with command routing and conversational fallback."""

import datetime
import json
import queue
import re
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus

import pyttsx3
import sounddevice as sd
import vosk

from brain import Brain
from data.dataset import load_dataset
from generate import generate_text


WAKE_WORDS = ("jarvis", "hey jarvis")
EXIT_WORDS = ("quit", "exit", "goodbye", "shutdown")
ACTIVE_WINDOW_SECONDS = 14
MEMORY_PATH = Path("data/operator_memory.json")

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


class JarvisAssistant:
    def __init__(self, model_path="vosk-model-small-en-us-0.15", samplerate=16000):
        self.engine = pyttsx3.init()
        self._configure_voice()

        self.model = vosk.Model(model_path)
        self.samplerate = samplerate
        self.recognizer = vosk.KaldiRecognizer(self.model, self.samplerate)
        self.audio_queue = queue.Queue()

        dataset = load_dataset()
        self.brain = Brain(dataset)

        self.sleep_mode = False
        self.history = []
        self.active_until = 0.0
        self.operator_memory = self._load_operator_memory()

    def _load_operator_memory(self):
        default_memory = {"operator_name": "", "notes": []}
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

        return {
            "operator_name": str(loaded.get("operator_name", "")).strip(),
            "notes": [str(note).strip() for note in notes if str(note).strip()],
        }

    def _save_operator_memory(self):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_PATH.open("w", encoding="utf-8") as file_handle:
            json.dump(self.operator_memory, file_handle, indent=2)

    def _activate_session(self):
        self.active_until = time.monotonic() + ACTIVE_WINDOW_SECONDS

    def _session_active(self):
        return time.monotonic() < self.active_until

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

    def _audio_callback(self, indata, frames, callback_time, status):
        del frames, callback_time
        if status:
            print(status)
        self.audio_queue.put(bytes(indata))

    def speak(self, text):
        print(f"Jarvis: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def _extract_directive(self, raw_text):
        text = raw_text.strip().lower()
        if not text:
            return "", False

        wake_used = False
        for wake in WAKE_WORDS:
            if text == wake:
                return "", True
            if text.startswith(wake + " "):
                return text[len(wake):].strip(" ,"), True

        if " jarvis " in f" {text} ":
            wake_used = True
            text = re.sub(r"\bhey\s+jarvis\b|\bjarvis\b", "", text).strip(" ,")

        return text, wake_used

    def _intent_response(self, text, wake_used):
        if any(self._contains_phrase(text, word) for word in EXIT_WORDS):
            return "__EXIT__"

        if "go to sleep" in text or "sleep mode" in text:
            self.sleep_mode = True
            self.active_until = 0.0
            return "Entering standby mode. Say Jarvis to wake me."

        if self.sleep_mode:
            if wake_used or "wake up" in text:
                self.sleep_mode = False
                self._activate_session()
                return "Online and listening."
            return None

        if text in {"", "listen", "are you there"} and wake_used:
            return "Listening."

        if re.search(r"\b(hello|hi|hey)\b", text):
            return "Good to hear from you. What do you need?"

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

        if "status report" in text or "system status" in text:
            return "All systems are stable. Voice input and response modules are operational."

        if "mission brief" in text:
            now_time = datetime.datetime.now().strftime("%I:%M %p")
            today = datetime.datetime.now().strftime("%A, %B %d")
            notes_count = len(self.operator_memory.get("notes", []))
            return (
                f"Mission brief: local time {now_time}, date {today}, "
                f"{notes_count} items stored in memory bank, and all assistant systems nominal."
            )

        if "time" in text:
            now_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now_time}."

        if "date" in text or "today" in text:
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
            return "I am JARVIS, your AI desktop assistant."

        if "help" in text:
            return (
                "Use wake phrase Jarvis, then ask for time, date, mission brief, open or close apps, "
                "search for a topic, or memory commands like remember that and what do you remember."
            )

        learned = self.brain.get_answer(text)
        if learned:
            return learned

        return None

    def _fallback_response(self, text):
        self.history.append(text)
        self.history = self.history[-20:]

        recent_context = " ".join(self.history[-4:])
        generated = generate_text(recent_context, length=16, temperature=0.8, top_k=25)

        cleaned = " ".join(generated.split())
        if not cleaned or len(cleaned) < 8:
            return "I can handle commands and quick questions. Say help to see what I can do."

        if cleaned[-1] not in ".!?":
            cleaned = cleaned + "."

        # Keep tone concise and command-center style.
        return f"Understood. {cleaned[0].upper() + cleaned[1:]}"

    def respond(self, raw_text):
        normalized, wake_used = self._extract_directive(raw_text)

        if wake_used:
            self._activate_session()

        if not wake_used and not self._session_active():
            return None

        if not normalized and wake_used:
            return "Listening."

        response = self._intent_response(normalized, wake_used)
        if response == "__EXIT__":
            return response
        if response:
            return response

        return self._fallback_response(normalized)

    def run(self):
        print(
            "Jarvis AI: online. Say 'Jarvis' to activate, then speak your command. "
            "Say 'quit' to exit."
        )

        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while True:
                data = self.audio_queue.get()
                if not self.recognizer.AcceptWaveform(data):
                    continue

                result = self.recognizer.Result()
                text = json.loads(result).get("text", "").strip()
                if not text:
                    continue

                print(f"You: {text}")
                response = self.respond(text)
                if not response:
                    continue

                if response == "__EXIT__":
                    self.speak("Powering down. Goodbye.")
                    break

                self.speak(response)


if __name__ == "__main__":
    assistant = JarvisAssistant()
    assistant.run()
                