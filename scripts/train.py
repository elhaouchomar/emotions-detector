import datetime
import os

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from preprocess import EMOTIONS, IMG_SIZE, preprocess_dataframe, BASE_DIR
from validation_loss_accuracy import plot_learning_curves

MODEL_DIR = os.path.join(BASE_DIR, "results/model")
DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")

ARCHITECTURE_NOTES = """Architecture notes
===================

Iteration 1: single conv block (32 filters) + dense(128) -> ~48% val accuracy,
             underfit, plateaued fast.
Iteration 2: 2 conv blocks (32, 64) with BatchNorm + Dropout(0.25) after each
             pool, dense(256) with Dropout(0.5) -> ~57% val accuracy, closed
             most of the gap to overfitting but still underfit slightly.
Iteration 3 (final): 3 conv blocks (32, 64, 128), BatchNorm + MaxPool +
             Dropout(0.25) after each block, GlobalAveragePooling instead of
             Flatten to cut parameter count, dense(256) + Dropout(0.5),
             softmax(7). This gave the best validation accuracy while
             keeping the train/val gap small. EarlyStopping (patience=8,
             restore_best_weights=True) on val_loss stopped training before
             overfitting set in; ModelCheckpoint kept the best epoch as a
             safety net.

Optimizer: Adam(lr=1e-3) with ReduceLROnPlateau(factor=0.5, patience=3) on
val_loss to fine-tune once progress stalls.
"""

DATA_AUGMENTATION = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08),
], name="data_augmentation")

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=len(EMOTIONS)):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            DATA_AUGMENTATION,

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.20),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.30),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.40),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.45),
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

    df = pd.read_csv(DATA_PATH)
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
        batch_size=32,
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

    print(f"Training complete. Check tensorboard logs at {os.path.join(MODEL_DIR, 'tensorboard_logs1')}")


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted. Exiting...")