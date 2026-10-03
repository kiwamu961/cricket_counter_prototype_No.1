import cv2
import time
import numpy as np
from hailo_platform import VDevice, FormatType

HEF_PATH = "/home/watanabe/yolov11s.hef"

CLASS_NAMES = [
    "Two Spotted Cricket F",
    "Two Spotted Cricket M",
    "House Cricket F",
    "House Cricket M",
]

INPUT_SIZE = 320
CONF_TH = 0.10
CAMERA_INDEX = 4


def draw_detections(frame, output_buffer):
    """Draw Hailo NMS output.
    Observed output format: first 2 values are header-like, then
    repeated [class_id, y1, x1, y2, x2, score]. Coordinates are normalized.
    """
    orig_h, orig_w = frame.shape[:2]
    out = output_buffer.flatten()
    data = out[2:]
    count = 0

    for i in range(0, len(data) - 5, 6):
        cls_id = int(data[i])
        y1 = float(data[i + 1])
        x1 = float(data[i + 2])
        y2 = float(data[i + 3])
        x2 = float(data[i + 4])
        score = float(data[i + 5])

        if cls_id == 0 and y1 == 0 and x1 == 0 and y2 == 0 and x2 == 0 and score == 0:
            break

        if score < CONF_TH:
            continue

        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue

        x1_i = int(x1 * orig_w)
        x2_i = int(x2 * orig_w)
        y1_i = int(y1 * orig_h)
        y2_i = int(y2 * orig_h)

        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1_i, max(y1_i - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        count += 1

    return count


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("camera open failed")
        return

    with VDevice() as target:
        infer_model = target.create_infer_model(HEF_PATH)
        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()
            input_buffer = np.empty((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            output_buffer = np.empty(infer_model.output().shape, dtype=np.float32)

            bindings.input().set_buffer(input_buffer)
            bindings.output().set_buffer(output_buffer)

            prev_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("frame read failed")
                    break

                # Adjust brightness for overexposed D435 image.
                frame_draw = cv2.convertScaleAbs(frame, alpha=0.7, beta=-40)

                rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
                input_buffer[0] = resized
                output_buffer.fill(0)

                configured_infer_model.run([bindings], timeout=10000)

                count = draw_detections(frame_draw, output_buffer)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                cv2.putText(
                    frame_draw,
                    f"FPS: {fps:.1f} Count: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("D435 + Hailo YOLO", frame_draw)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
