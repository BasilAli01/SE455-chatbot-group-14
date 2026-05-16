# Multimodal IoT Scene Narrator

**Generative AI Course Project — Multimodal IoT Systems with Vision-Language Models**

A real-time scene narration system that combines **YOLOv8** object detection with **GPT-4V** vision-language understanding to generate audio descriptions of the user's environment. Built as a proof-of-concept for the **Beyond Vision** assistive eyeglasses project.

---

## How It Works

```
Webcam → YOLOv8 (object detection) → GPT-4V (scene narration) → TTS (audio output)
```

1. **Camera captures** live frames from your laptop webcam
2. **YOLOv8** detects and labels objects in real-time (~30 FPS)
3. **GPT-4V** receives the frame + YOLO detections and generates a natural-language scene description (every 5 seconds)
4. **TTS** speaks the narration aloud

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-your-key-here"

# Windows
set OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

```bash
python main.py
```

---

## Controls

| Key | Action                        |
| --- | ----------------------------- |
| `q` | Quit                          |
| `s` | Force scene narration now     |
| `m` | Mute / Unmute audio           |
| `l` | Show narration log in console |

---

## Configuration

Edit the `Config` class in `scene_narrator.py`:

| Parameter            | Default    | Description                    |
| -------------------- | ---------- | ------------------------------ |
| `CAMERA_INDEX`       | 0          | Webcam index                   |
| `YOLO_MODEL`         | yolov8n.pt | YOLO model (n/s/m/l/x)         |
| `YOLO_CONFIDENCE`    | 0.40       | Min detection confidence       |
| `NARRATION_INTERVAL` | 5          | Seconds between VLM calls      |
| `GPT_MODEL`          | gpt-4o     | OpenAI model with vision       |
| `USE_OPENAI_TTS`     | True       | True=OpenAI TTS, False=pyttsx3 |

---

## Without an API Key

The system works without an OpenAI API key in **degraded mode**:

- YOLO still runs and detects objects
- Narrations fall back to simple detection summaries
- pyttsx3 provides offline TTS

---

## Project Structure

```
SE455-chatbot-group-14/
├── main.py              # App entry point and main loop
├── config.py            # All configuration and parameters
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .env.example         # API key template
├── core/
│   ├── detector.py      # YOLODetector (YOLOv8)
│   ├── narrator.py      # SceneNarrator (GPT-4V)
│   └── tts.py           # TTSEngine (OpenAI TTS + pyttsx3)
├── tests/               # Unit tests
└── logs/                # Auto-generated narration logs```

---

## Connection to Beyond Vision

This project maps directly to Beyond Vision's architecture:

| Scene Narrator       | Beyond Vision               |
| -------------------- | --------------------------- |
| Laptop webcam        | ArduCam IMX477 cameras      |
| YOLOv8 on CPU        | YOLO on Hailo-8 accelerator |
| GPT-4V via API       | On-device VLM (future)      |
| OpenAI TTS / pyttsx3 | Sherpa-ONNX TTS             |
| Laptop speakers      | Bone conduction speakers    |

---

## Team

- Tariq Dabbagh 230279
- Abdulrahman Mahmalji 230474
- Abdullah Damati 230729
- Basil Ali 230653
- Abdulaziz Albaz 230361

**Supervisor:** Dr. Nidal Nasser
