import os
import re

path = 'app/jarvis.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'    def __init__\(self, model_path=\'vosk-model-small-en-us-0\.15\', samplerate=16000\):.*?(?=\s+def _configure_voice\(self\):)',
'''    def __init__(self):
        self.engine = pyttsx3.init()
        self._configure_voice()
        self.recognizer = sr.Recognizer()

        from core.inference import InferenceManager
        from core.memory import MemoryManager
        from services.task_services import TaskService

        self.memory_manager = MemoryManager()
        self.task_service = TaskService()
        self.inference_manager = InferenceManager(self)
        self.sleep_mode = False

        self.operator_memory = self.memory_manager.operator_memory
        self.user_preferences = self.memory_manager.user_preferences
        self.daily_planner = self.task_service.daily_planner
        self.session_memory = self.memory_manager.session_memory

        self.conversation_state = {
            "last_topic": "", "last_user_text": "", "last_assistant_text": "",
            "turn_count": 0, "last_user_mood": "neutral", "last_goal": "",
            "last_knowledge_intent": "", "last_knowledge_subject": "",
            "last_plan_goal": "", "last_plan_steps": []
        }

        self.knowledge_cache = {}
        self.active_timers = []
        self._last_reminder_tick = 0.0
        self._last_handled_text = ""
        self._last_handled_at = 0.0
        self.active_until = 0.0
''', text, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
