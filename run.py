import speech_recognition as sr
from app.jarvis import JarvisAssistant
from app.friday import FridayAssistant
import sys


def _route_target(text, jarvis, friday):
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return jarvis, ""

    if lowered.startswith("switch to friday") or lowered == "friday mode":
        return friday, lowered.replace("switch to friday", "").strip()

    if lowered.startswith("switch to jarvis") or lowered == "jarvis mode":
        return jarvis, lowered.replace("switch to jarvis", "").strip()

    if lowered.startswith(("friday ", "hey friday ", "fry day ", "fri day ")) or lowered == "friday":
        cleaned = lowered
        for wake in ("hey friday", "fry day", "fri day", "friday"):
            if cleaned == wake:
                return friday, ""
            if cleaned.startswith(wake + " "):
                return friday, cleaned[len(wake):].strip(" ,")
        return friday, lowered

    if lowered.startswith(("jarvis ", "hey jarvis ", "jervis ", "jarves ", "jarvice ", "javis ")) or lowered == "jarvis":
        cleaned = lowered
        for wake in ("hey jarvis", "jervis", "jarves", "jarvice", "javis", "jarvis"):
            if cleaned == wake:
                return jarvis, ""
            if cleaned.startswith(wake + " "):
                return jarvis, cleaned[len(wake):].strip(" ,")
        return jarvis, lowered

    return jarvis, lowered


def _set_conversational_mode(enabled, jarvis, friday):
    jarvis.set_conversational_mode(enabled)
    friday.set_conversational_mode(enabled)

def main():
    print("Initializing Master Controller...")
    
    print("Loading JARVIS...")
    jarvis = JarvisAssistant()
    
    print("Loading FRIDAY...")
    friday = FridayAssistant()
    
    # Intialize shared resources
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("Calibrating background noise...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        recognizer.dynamic_energy_threshold = True
        
    jarvis.speak("Master system online. Both JARVIS and FRIDAY are listening simultaneously.")
    print("\nMaster system online.")

    # Start with Jarvis implicitly active
    active_ai = jarvis
    
    while True:
        try:
            # Check timers for both instances
            for ai in [jarvis, friday]:
                if hasattr(ai, '_check_due_timers'):
                    alerts = ai._check_due_timers()
                    if hasattr(ai, '_check_due_reminders'):
                        alerts.extend(ai._check_due_reminders())
                    for alert in alerts:
                        ai.speak(alert)

            with microphone as source:
                # We listen silently, avoiding printing "Listening..." repeatedly causing spam
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

            # Convert to text
            text = recognizer.recognize_google(audio).lower()
            print(f"\nYou: {text}")

            # Routing Logic (one utterance goes to one assistant)
            active_ai, routed_text = _route_target(text, jarvis, friday)
            if active_ai is jarvis:
                print("[Routed to JARVIS]")
            else:
                print("[Routed to FRIDAY]")

            if (
                text.strip() == "conversational mode"
                or "conversational mode on" in text
                or "enable conversational mode" in text
            ):
                _set_conversational_mode(True, jarvis, friday)
            elif "conversational mode off" in text or "disable conversational mode" in text:
                _set_conversational_mode(False, jarvis, friday)
            
            # Prevent double-processing the same phrase using the cleaned command
            if active_ai._is_duplicate_utterance(routed_text or text):
                continue

            # Pass the original text (including wake word) so the assistant's internal
            # wake-word extraction recognizes the directive and activates the session.
            response = active_ai.respond(text)
            
            if response == '__EXIT__':
                active_ai.speak("Powering down Master Controller. Goodbye.")
                break
                
            if response:
                active_ai.speak(response)

        except sr.WaitTimeoutError:
            # Re-starts the loop silently without printing "Listening..." infinite times
            continue
        except sr.UnknownValueError:
            # Could not understand audio, restart loop silently
            continue
        except sr.RequestError as e:
            print(f"Google SR error: {e}")
        except KeyboardInterrupt:
            print("\nShutting down master controller...")
            break
        except Exception as e:
            print(f"Runtime error: {e}")

if __name__ == "__main__":
    main()
