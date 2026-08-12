from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="models/yolov8n.pt"):

        self.model = YOLO(model_path)

        self.animal_classes = {
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe"
        }

    def detect(self, frame):

        detections = []

        results = self.model(
            frame,
            verbose=False,
            conf=0.25,
            imgsz=416
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls]

                if label == "person":
                    category = "HUMAN"

                elif label in self.animal_classes:
                    category = "ANIMAL"

                else:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "category": category,
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections