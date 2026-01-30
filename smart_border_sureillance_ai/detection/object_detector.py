from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        detections = []
        results = self.model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]

                if label == "person":
                    category = "HUMAN"
                elif label in ["dog", "cow", "horse", "sheep"]:
                    category = "ANIMAL"
                else:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "category": category,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
