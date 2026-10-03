import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import serial

model = YOLO("best_1.pt")
ser = serial.Serial('COM3', 9600)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())

        results = model(img, conf=0.5)

        futa_male = 0
        futa_female = 0
        ie_male = 0
        ie_female = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]

            if "Two Spotted Cricket" in label:
                if label.endswith("M"):
                    futa_male += 1
                elif label.endswith("F"):
                    futa_female += 1
            elif "House Cricket" in label:
                if label.endswith("F"):
                    ie_male += 1
                elif label.endswith("F"):
                    ie_female += 1

        data = f"{futa_male},{futa_female},{ie_male},{ie_female}\n"
        ser.write(data.encode())

        annotated = results[0].plot()

        cv2.putText(annotated, f"Futa M: {futa_male}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated, f"Futa F: {futa_female}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated, f"Ie M: {ie_male}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Ie F: {ie_female}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cricket Detection", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()           
    ser.close 