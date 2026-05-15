import cv2
from config import Config

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. YOLO detection disabled.")
    print("       Install with: pip install ultralytics")


class YOLODetector:
    def __init__(self):
        if not YOLO_AVAILABLE:
            self.model = None
            return
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(Config.YOLO_MODEL)
        print("[INFO] YOLO model loaded.")

    def detect(self, frame):
        if self.model is None:
            return []

        results = self.model(frame, conf=Config.YOLO_CONFIDENCE, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections

    def format_detections(self, detections):
        if not detections:
            return "No objects detected by YOLO."

        groups = {}
        for d in detections:
            cls = d["class"]
            if cls not in groups:
                groups[cls] = []
            groups[cls].append(d["confidence"])

        parts = []
        for cls, confs in groups.items():
            conf_str = ", ".join(f"{c:.2f}" for c in confs)
            parts.append(f"{len(confs)} {cls}(s) [conf: {conf_str}]")

        return "YOLO detections: " + "; ".join(parts)

    def draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['class']} {d['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame
