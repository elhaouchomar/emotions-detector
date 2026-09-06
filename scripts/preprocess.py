import argparse, os, cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = 48
EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def preprocess_dataframe(df):
    X = np.stack(df["pixels"].apply(lambda s: np.fromstring(s, sep=" ", dtype=np.float32)))
    return X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0, df["emotion"].to_numpy()


def detect_and_crop_face(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(48, 48))
    if len(faces) == 0: return None
    x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
    return cv2.resize(gray[y:y+h,x:x+w], (48,48)), (x,y,w,h)


def predict_emotion(face, model):
    p = model(face.reshape(1,48,48,1).astype("float32")/255.0, training=False).numpy()[0]
    i = np.argmax(p)
    return f"{EMOTIONS[i]} ({p[i]*100:.0f}%)"


def draw_bounding_box_with_emotion(frame, box, emotion):
    if box is not None and emotion:
        x,y,w,h = box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,emotion,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,0),2)


def run_preprocessing_test(video_path, out_dir, seconds=20):
    from tensorflow import keras
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    model = keras.models.load_model(
        os.path.join(BASE_DIR,"results/model/final_emotion_model.keras")
    )

    frame_idx = saved = 0
    box = emotion = None

    while frame_idx < seconds * fps:
        ok, frame = cap.read()
        if not ok: break

        if frame_idx % max(int(fps),1) == 0:
            result = detect_and_crop_face(frame,cascade)
            if result:
                face,box = result
                emotion = predict_emotion(face,model)
                cv2.imwrite(os.path.join(out_dir,f"image{saved}.png"),face)
                print(f"{saved+1}s: {emotion}")
                saved += 1

        draw_bounding_box_with_emotion(frame,box,emotion)
        cv2.imshow("Preprocessing Test",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",default=os.path.join(BASE_DIR,"results/preprocessing_test/input_video.mp4"))
    p.add_argument("--out",default=os.path.join(BASE_DIR,"results/preprocessing_test"))
    p.add_argument("--seconds",type=int,default=20)
    a = p.parse_args()
    run_preprocessing_test(a.video,a.out,a.seconds)


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")