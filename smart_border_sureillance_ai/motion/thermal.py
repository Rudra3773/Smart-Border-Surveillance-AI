import cv2

def thermal_view(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal
