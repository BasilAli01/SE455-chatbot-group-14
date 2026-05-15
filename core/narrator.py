import cv2
import base64
from config import Config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARN] openai not installed. GPT-4V narration disabled.")
    print("       Install with: pip install openai")


class SceneNarrator:
    def __init__(self):
        self.client = None
        if OPENAI_AVAILABLE and Config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            print("[INFO] OpenAI client initialized.")
        else:
            if not Config.OPENAI_API_KEY:
                print("[WARN] OPENAI_API_KEY not set. Set it in your .env file.")
            print("[INFO] Scene narration will use mock responses.")

    def encode_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def narrate(self, frame, yolo_summary):
        if self.client is None:
            return self._mock_narrate(yolo_summary)

        try:
            base64_image = self.encode_frame(frame)

            response = self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                max_tokens=Config.MAX_TOKENS,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{yolo_summary}\n\nDescribe what you see for a visually impaired user."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[ERROR] GPT-4V API error: {e}")
            return f"Scene narration unavailable. {yolo_summary}"

    def _mock_narrate(self, yolo_summary):
        if "No objects" in yolo_summary:
            return "The area ahead appears clear. No notable objects detected."
        return f"Objects detected in your environment. {yolo_summary}"
