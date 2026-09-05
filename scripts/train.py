import datetime
import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import EMOTIONS, IMG_SIZE, load_csv, preprocess_dataframe, BASE_DIR
from validation_loss_accuracy import plot_learning_curves

MODEL_DIR = os.path.join(BASE_DIR, "results/model")
DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")

ARCHITECTURE_NOTES = "Arch: 3 conv blocks, GlobalAveragePooling, Dense(256), Softmax(7). Opt: Adam(1e-3)."


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=len(EMOTIONS)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_csv(DATA_PATH)
    X, y = preprocess_dataframe(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    model = build_model()

    log_dir = os.path.join(
        MODEL_DIR, "tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "final_emotion_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
    )

    model.save(os.path.join(MODEL_DIR, "final_emotion_model.keras"))

    arch_path = os.path.join(MODEL_DIR, "final_emotion_model_arch.txt")
    with open(arch_path, "w") as f:
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        f.write("\n".join(summary_lines))
        f.write("\n\n")
        f.write(ARCHITECTURE_NOTES)
    print(f"Saved architecture summary to {arch_path}")

    plot_learning_curves(history, os.path.join(MODEL_DIR, "learning_curves.png"))

    print(f"Training complete. Check tensorboard logs at {os.path.join(MODEL_DIR, 'tensorboard_logs')}")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")
