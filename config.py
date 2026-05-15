import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    # Camera
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # YOLO
    YOLO_MODEL = "models/yolov8n.pt"
    YOLO_CONFIDENCE = 0.40

    # GPT-4V
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GPT_MODEL = "gpt-4o"
    NARRATION_INTERVAL = 5
    MAX_TOKENS = 200

    # TTS
    USE_OPENAI_TTS = True
    TTS_VOICE = "nova"
    TTS_SPEED = 1.0

    # Paths
    LOG_FILE = "logs/narration_log.json"

    # System prompt
    SYSTEM_PROMPT = (
        "You are a scene narrator for a visually impaired person. "
        "You receive a camera frame and a list of objects detected by YOLO. "
        "Narrate what the person needs to know in 2-3 SHORT sentences, "
        "like a GPS device would. Focus on: spatial layout, notable objects, "
        "people, and any potential hazards. Be concise and factual. "
        "Do NOT start with 'I see' or 'The image shows'. "
        "Speak directly to the user as if guiding them."
    )
