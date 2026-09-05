# Emotions Detector

Real-time facial emotion recognition: a CNN trained on the FER2013-style
Kaggle dataset, plugged into an OpenCV webcam pipeline that detects faces,
crops/normalizes them to 48x48 grayscale, and predicts one of 7 emotions
(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) at least once per second.

## Project structure

```
project
├── data
│   ├── train.csv
│   ├── test_with_emotions.csv
│   └── test.csv
├── requirements.txt
├── README.md
├── results
│   ├── model
│   │   ├── learning_curves.png
│   │   ├── final_emotion_model_arch.txt
│   │   ├── final_emotion_model.keras
│   └── preprocessing_test
│       ├── image0.png ... image_n.png
│       └── input_video.mp4
└── scripts
    ├── preprocess.py            # image/CSV loading, face detection, 48x48 grayscale pipeline
    ├── train.py                 # builds + trains the CNN, saves model/arch/curves/tensorboard
    ├── validation_loss_accuracy.py  # plots learning curves from a Keras History object
    ├── predict.py                # evaluates final_emotion_model.keras on the test set
    └──predict_live_stream.py    # webcam (or recorded video) -> live emotion prediction

```

## 1. Get the data

Download the dataset from the Kaggle "Facial Expression Recognition Challenge"
page (fer2013). You need `train.csv` and `test_with_emotions.csv` (`emotion`,
`pixels`, `usage` columns — pixels are a space-separated 48x48=2304 int string).
Place both files in `data/`.

## 2. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 3. Train the classifier

```bash
python ./scripts/train.py
```

This will:
- Load and preprocess `data/train.csv` (normalize, reshape to 48x48x1, one-hot labels)
- Split off a validation set from the training data (test set is never used for fitting)
- Train a CNN with `EarlyStopping`, `ModelCheckpoint`, and a `TensorBoard` callback
  (logs written to `results/model/tensorboard_logs/`)
- Save the trained model to `results/model/final_emotion_model.keras`
- Save `model.summary()` plus a short explanation of the architecture and the
  iterations that led to it in `results/model/final_emotion_model_arch.txt`
- Save the learning curves (loss + accuracy, train vs val) to
  `results/model/learning_curves.png`

To capture `results/model/tensorboard.png`, run `tensorboard --logdir
results/model/tensorboard_logs` during/after training and screenshot the UI.

## 4. Evaluate on the test set

```bash
python ./scripts/predict.py
```

Expected output:
```
Accuracy on test set: 62%
```

## 5. Preprocessing pipeline test (face detection)

```bash
python ./scripts/preprocess.py --video path/to/20s_face_video.mp4 --out results/preprocessing_test
```

Reads a >=20s video containing a face, detects the face each second with
OpenCV's Haar cascade (`cv2.CascadeClassifier`), crops/centers it, resizes to
48x48 grayscale, and writes `image0.png ... image_n.png` (20-21 images) plus a
copy of the input as `input_video.mp4`.

## 6. Live webcam prediction

```bash
python ./scripts/predict_live_stream.py
```

Expected output:
```
Reading video stream ...

Preprocessing ...
11:11:11s : Happy , 73%

Preprocessing ...
11:11:12s : Happy , 93%
...
```

If no webcam is available, pass a recorded video instead:

```bash
python ./scripts/predict_live_stream.py --video path/to/video.mp4
```

## 7. (Optional) Hack the CNN

```bash
python ./scripts/hack_cnn.py --image path/to/happy_image.png
```

Starts from an image the model classifies as `Happy` with >90% probability
and nudges pixel values (gradient-based, weights frozen) until the model
predicts `Sad`, while keeping the perturbation visually unnoticeable.

## Notes

- The model is only ever fit on `train.csv`; `test_with_emotions.csv` is used
  strictly for evaluation.
- Training is stopped by `EarlyStopping` before the validation loss starts
  climbing (see `learning_curves.png`), so the saved model is the
  pre-overfitting checkpoint, not the last epoch.
