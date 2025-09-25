import cv2
import numpy as np
import time
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection: " + cascade_path)
    return face_cascade


def draw_label(frame: np.ndarray, text: str, left: int, top: int, color: tuple) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (left, top - text_height - baseline - 6), (left + text_width + 6, top), color, -1)
    cv2.putText(frame, text, (left + 3, top - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    model_path = "mask_detector.h5"

    try:
        model = load_model(model_path)
    except Exception as exc:
        print(f"[ERROR] Cannot load model '{model_path}': {exc}")
        sys.exit(1)

    try:
        face_cascade = load_face_detector()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("[INFO] Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Ensure a camera is connected and not in use.")
        sys.exit(1)

    # For FPS calculation
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame from webcam.")
                break

            # Convert to grayscale for detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_array = np.asarray(face_resized, dtype=np.float32)
                face_array = preprocess_input(face_array)
                face_array = np.expand_dims(face_array, axis=0)

                preds = model.predict(face_array, verbose=0)[0]
                (mask_prob, no_mask_prob) = preds
                label = "With Mask" if mask_prob > no_mask_prob else "No Mask"
                color = (0, 200, 0) if label == "With Mask" else (0, 0, 255)
                confidence = max(mask_prob, no_mask_prob) * 100.0

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, f"{label}: {confidence:.1f}%", x, y, color)

            # FPS
            curr_time = time.time()
            if curr_time - prev_time > 0:
                fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Mask Detection (press 'q' to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


