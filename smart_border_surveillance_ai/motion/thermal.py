import cv2
import numpy as np


def is_night(frame, threshold=75):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    return brightness < threshold


def thermal_view(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Improve low-light visibility
    gray = cv2.equalizeHist(gray)

    thermal = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    return thermal