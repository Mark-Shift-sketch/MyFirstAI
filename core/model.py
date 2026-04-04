import queue
import vosk
try:
    import speech_recognition as sr
except Exception:
    sr = None

from models.RAG.brain import Brain
from core.data_service import load_dataset
from models.RAG.generator import LocalInstructionGenerator

class ModelManager:
    def __init__(self, model_path="vosk-model-small-en-us-0.15", samplerate=16000):
        self.model_path = model_path
        self.samplerate = samplerate
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.samplerate)
        self.audio_queue = queue.Queue()
        self.online_recognizer = sr.Recognizer() if sr is not None else None
        self.online_stt_enabled = self.online_recognizer is not None
        self.online_stt_failures = 0
        
        self.dataset = load_dataset()
        self.brain = Brain(self.dataset)
        self.rag_generator = LocalInstructionGenerator()
        
    def reload_nlp_models(self):
        self.dataset = load_dataset()
        self.brain = Brain(self.dataset)
