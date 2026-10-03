import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "/home/watanabe/best_1.pt"
CAMERA_INDEX = 2
CONF_TH = 0.5
IMG_SIZE = 320

model = YOLO(MODEL_PATH)
print("model names:", model.names)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    print("camera open failed")
    raise SystemExit

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("camera read failed")
        break

    result = model(frame, conf=CONF_TH, imgsz=IMG_SIZE, device="cpu", verbose=False)[0]
    img = result.plot()

    counts = {"FutaM": 0, "FutaF": 0, "IeM": 0, "IeF": 0}

    for box in result.boxes:
        cls = int(box.cls[0])
        label = result.names[cls]

        if label == "Two Spotted Cricket M":
            counts["FutaM"] += 1
        elif label == "Two Spotted Cricket F":
            counts["FutaF"] += 1
        elif label == "House Cricket M":
            counts["IeM"] += 1
        elif label == "House Cricket F":
            counts["IeF"] += 1

    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now

    cv2.putText(img, f"FPS:{fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"FutaM:{counts['FutaM']}  FutaF:{counts['FutaF']}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"IeM:{counts['IeM']}  IeF:{counts['IeF']}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("CPU YOLO Count", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
