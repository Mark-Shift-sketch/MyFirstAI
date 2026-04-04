import os
import re

path = 'app/jarvis.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'    def run\(self\):.*?(?=if __name__ == "__main__":)',
'''    def run(self):
        self.speak('System online Sir. Say ''quit'' to exit.')
        self.microphone = sr.Microphone()
        
        while True:
            try:
                if hasattr(self.task_service, '_check_due_timers'):
                    alerts = self.task_service._check_due_timers()
                    alerts.extend(self.task_service._check_due_reminders())
                    for alert in alerts:
                        self.speak(alert)

                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                
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
                
''', text, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
