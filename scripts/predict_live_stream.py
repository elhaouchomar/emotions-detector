import argparse
import datetime
import os
import time

import cv2
from tensorflow import keras
from preprocess import FACE_CASCADE_PATH, detect_and_crop_face, predict_emotion, BASE_DIR, draw_bounding_box_with_emotion

MODEL_PATH = os.path.join(BASE_DIR, "results/model/final_emotion_model.keras")


def open_capture(video_path: str):
    cap = cv2.VideoCapture(video_path or 0)
    if cap.isOpened(): return cap
    fallback = os.path.join(BASE_DIR, "data/fallback_video.mp4")
    if os.path.exists(fallback): return cv2.VideoCapture(fallback)
    raise RuntimeError("No webcam or video provided.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Video path")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train.py first.")

    model = keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = open_capture(args.video)
    print("Reading video stream ...\n")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(round(fps)), 1)
    frame_idx = 0

    try:
        latest_emotion = ""
        latest_box = None
        print("Reading video stream ...")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                print("Preprocessing ...")
                result = detect_and_crop_face(frame, face_cascade)
                if result is not None:
                    face, latest_box = result
                    latest_emotion = predict_emotion(face, model)
                    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}s : {latest_emotion}\n")
                else:
                    latest_box, latest_emotion = None, ""
                if not args.video:
                    time.sleep(max(0, 1 - 0.0))

            draw_bounding_box_with_emotion(frame, latest_box, latest_emotion)
            
            cv2.imshow("Live Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")
