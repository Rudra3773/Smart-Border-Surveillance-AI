import cv2

class MotionDetector:
    def __init__(self, threshold=5000):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.threshold = threshold

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)
        return motion_pixels > self.threshold
