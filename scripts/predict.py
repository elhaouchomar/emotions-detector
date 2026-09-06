import os

from tensorflow import keras
import pandas as pd

from preprocess import preprocess_dataframe, BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "results/model/final_emotion_model.keras")
TEST_PATH = os.path.join(BASE_DIR, "data/test_with_emotions.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train.py first.")

    model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    )

    df = pd.read_csv(TEST_PATH)
    X_test, y_test = preprocess_dataframe(df)

    model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    )


    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy on test set: {accuracy * 100:.3f}%")
    print(f"Loss on test set: {loss:.3f}")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")