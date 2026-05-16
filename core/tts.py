import os
import sys
import threading
from config import Config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[WARN] pyttsx3 not installed. Offline TTS disabled.")
    print("       Install with: pip install pyttsx3")


class TTSEngine:
    def __init__(self):
        self.openai_client = None
        self.pyttsx3_engine = None
        self.is_speaking = False
        self.muted = False

        if Config.USE_OPENAI_TTS and OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            print("[INFO] TTS: Using OpenAI TTS API.")
        elif PYTTSX3_AVAILABLE:
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', 160)
            print("[INFO] TTS: Using pyttsx3 offline engine.")
        else:
            print("[WARN] No TTS engine available. Audio output disabled.")

    def speak(self, text):
        if self.muted or not text:
            return
        thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        thread.start()

    def _speak_sync(self, text):
        if self.is_speaking:
            return
        self.is_speaking = True
        try:
            if self.openai_client:
                self._speak_openai(text)
            elif self.pyttsx3_engine:
                self._speak_pyttsx3(text)
            else:
                print(f"[TTS] {text}")
        finally:
            self.is_speaking = False

    def _speak_openai(self, text):
    try:
        import tempfile
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice=Config.TTS_VOICE,
            input=text,
            speed=Config.TTS_SPEED
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        if sys.platform == "darwin":
            os.system(f"afplay {temp_path} &")
        elif sys.platform == "linux":
            os.system(f"mpg123 -q {temp_path} 2>/dev/null || ffplay -nodisp -autoexit {temp_path} 2>/dev/null &")
        else:
            os.startfile(temp_path)

    except Exception as e:
        print(f"[ERROR] OpenAI TTS failed: {e}")
        if PYTTSX3_AVAILABLE:
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] pyttsx3 TTS failed: {e}")

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted
