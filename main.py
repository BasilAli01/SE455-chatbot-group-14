"""
Multimodal IoT Scene Narrator
==============================
Integrates YOLO object detection with GPT-4V vision-language model
to generate real-time audio scene descriptions from a laptop camera.

Project: Generative AI Course - Multimodal IoT Systems with Vision-Language Models
Link to: Beyond Vision (Environmental Awareness Module proof-of-concept)

Usage:
    python main.py

    Press 'q' to quit, 's' to force a scene narration, 'm' to mute/unmute.
"""

import cv2
import time
import threading
import json
import os
from datetime import datetime

from config import Config
from core import YOLODetector, SceneNarrator, TTSEngine


class SceneNarratorApp:
    def __init__(self):
        print("=" * 60)
        print("  Multimodal IoT Scene Narrator")
        print("  YOLO + GPT-4V + TTS Pipeline")
        print("=" * 60)
        print()

        self.detector = YOLODetector()
        self.narrator = SceneNarrator()
        self.tts = TTSEngine()

        self.last_narration_time = 0
        self.last_narration_text = ""
        self.last_narration_latency = None
        self.latest_detections = []
        self.narration_in_progress = False
        self.running = True

        self.log_entries = []
        os.makedirs("logs", exist_ok=True)

    def run(self):
        print()
        print("[INFO] Opening camera...")
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera. Check CAMERA_INDEX in Config.")
            print("[INFO] Running in demo mode with a blank frame...")
            self._run_demo_mode()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        print("[INFO] Camera opened successfully.")
        print()
        print("Controls:")
        print("  q - Quit")
        print("  s - Force scene narration now")
        print("  m - Mute/Unmute audio")
        print("  l - Show detection log")
        print()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Frame capture failed. Retrying...")
                    continue

                detections = self.detector.detect(frame)
                self.latest_detections = detections

                display_frame = frame.copy()
                display_frame = self.detector.draw_detections(display_frame, detections)
                self._draw_status(display_frame)

                current_time = time.time()
                if current_time - self.last_narration_time >= Config.NARRATION_INTERVAL and not self.narration_in_progress:
                    self._trigger_narration(frame.copy(), detections)

                cv2.imshow("Scene Narrator - YOLO + GPT-4V", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._trigger_narration(frame.copy(), detections, force=True)
                elif key == ord('m'):
                    muted = self.tts.toggle_mute()
                    print(f"[INFO] Audio {'muted' if muted else 'unmuted'}.")
                elif key == ord('l'):
                    self._print_log()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._save_log()
            print("[INFO] Application stopped.")

    def _run_demo_mode(self):
        print()
        print("=" * 50)
        print("  DEMO MODE (no camera)")
        print("=" * 50)
        print()

        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Demo Mode - No Camera", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        detections = self.detector.detect(dummy_frame)
        yolo_summary = self.detector.format_detections(detections)
        print(f"YOLO: {yolo_summary}")

        print("Generating scene narration...")
        narration = self.narrator.narrate(dummy_frame, yolo_summary)
        print(f"Narration: {narration}")

        print("Speaking narration...")
        self.tts.speak(narration)
        time.sleep(5)

        print()
        print("[INFO] Demo complete. To use with a camera, ensure a webcam is connected.")

    def _trigger_narration(self, frame, detections, force=False):
        if self.narration_in_progress and not force:
            return

        self.narration_in_progress = True
        yolo_summary = self.detector.format_detections(detections)

        def _narrate():
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Generating narration...")
                print(f"  YOLO: {yolo_summary}")

                t0 = time.time()
                narration = self.narrator.narrate(frame, yolo_summary)
                latency = time.time() - t0

                print(f"  Latency: {latency:.2f}s")
                print(f"  Narration: {narration}")

                self.last_narration_text = narration
                self.last_narration_latency = latency
                self.last_narration_time = time.time()

                self.log_entries.append({
                    "timestamp": timestamp,
                    "detections": len(detections),
                    "yolo_summary": yolo_summary,
                    "latency_s": round(latency, 2),
                    "narration": narration
                })

                self.tts.speak(narration)
                self._save_log()

            finally:
                self.narration_in_progress = False

        thread = threading.Thread(target=_narrate, daemon=True)
        thread.start()

    def _draw_status(self, frame):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        n = len(self.latest_detections)
        cv2.putText(frame, f"Objects: {n}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elapsed = time.time() - self.last_narration_time
        remaining = max(0, Config.NARRATION_INTERVAL - elapsed)
        cv2.putText(frame, f"Next narration: {remaining:.0f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if self.last_narration_latency is not None:
            cv2.putText(frame, f"Latency: {self.last_narration_latency:.2f}s", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        mute_text = "MUTED" if self.tts.muted else "AUDIO ON"
        mute_color = (0, 0, 255) if self.tts.muted else (0, 255, 0)
        cv2.putText(frame, mute_text, (w - 120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mute_color, 1)

        cv2.putText(frame, "q:Quit  s:Narrate  m:Mute", (w - 280, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if self.last_narration_text:
            words = self.last_narration_text[:120]
            if len(self.last_narration_text) > 120:
                words += "..."
            cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, words, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _print_log(self):
        print("\n" + "=" * 50)
        print("  NARRATION LOG")
        print("=" * 50)
        for entry in self.log_entries[-10:]:
            print(f"  [{entry['timestamp']}] Objects: {entry['detections']} | Latency: {entry.get('latency_s', '?')}s")
            print(f"    {entry['narration']}")
        print("=" * 50 + "\n")

    def _save_log(self):
        if self.log_entries:
            with open(Config.LOG_FILE, 'w') as f:
                json.dump(self.log_entries, f, indent=2)


if __name__ == "__main__":
    app = SceneNarratorApp()
    app.run()
