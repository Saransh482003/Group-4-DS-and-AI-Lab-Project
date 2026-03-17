from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = Path(__file__).with_name("best.pt")
CAMERA_INDEX = 1
CONFIDENCE = 0.70
WINDOW_NAME = "YOLO Inference"

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(CAMERA_INDEX)

print("Running inference. Press 'q' or ESC to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
