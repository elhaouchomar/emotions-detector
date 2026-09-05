import argparse
import os

import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def preprocess_dataframe(df: pd.DataFrame):
    pixels = df["pixels"].apply(
        lambda s: np.fromstring(s, sep=" ", dtype=np.float32)
    )
    X = np.stack(pixels.values).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = df["emotion"].to_numpy()
    return X, y


def detect_and_crop_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return face, (x, y, w, h)


def predict_emotion(face, model):
    if model is None: return ""
    X = face.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
    probs = model(X, training=False).numpy()[0]
    return f"{EMOTIONS[int(np.argmax(probs))]} ({np.max(probs)*100:.0f}%)"


def draw_bounding_box_with_emotion(frame, box, emotion):
    if box is not None and emotion:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def run_preprocessing_test(video_path: str, out_dir: str, seconds: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps < 1:
        fps = 25.0
    frame_interval = max(int(round(fps)), 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        width, height = 640, 480

    out_video_path = os.path.join(out_dir, "output_video.mp4")
    
    video_writer = None
    if not video_path or os.path.abspath(video_path) != os.path.abspath(out_video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    model_path = os.path.join(BASE_DIR, "results/model/final_emotion_model.keras")
    model = None
    if os.path.exists(model_path):
        from tensorflow import keras
        model = keras.models.load_model(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Cannot predict emotions.")

    saved = 0
    frame_idx = 0
    latest_box = None
    latest_emotion = ""
    
    while saved <= seconds:
        ok, frame = cap.read()
        if not ok:
            break
            
        if frame_idx % frame_interval == 0:
            result = detect_and_crop_face(frame, face_cascade)
            if result is not None:
                face, box = result
                cv2.imwrite(os.path.join(out_dir, f"image{saved}.png"), face)
                saved += 1
                latest_emotion = predict_emotion(face, model)
                latest_box = box
                print(f"Frame {frame_idx}: {latest_emotion}")
            else:
                latest_box = None
                latest_emotion = ""
                
        draw_bounding_box_with_emotion(frame, latest_box, latest_emotion)
            
        if video_writer is not None:
            video_writer.write(frame)
            
        cv2.imshow("Preprocessing Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1
        
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved {saved} face crops to {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=os.path.join(BASE_DIR, "results/preprocessing_test/input_video.mp4"),
        help="Video path (empty for webcam)",
    )
    parser.add_argument(
        "--out", default=os.path.join(BASE_DIR, "results/preprocessing_test"), help="Out dir"
    )
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    try: run_preprocessing_test(args.video, args.out, args.seconds)
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")
