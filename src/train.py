import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "data" / "number_model.keras"

IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 10

def main():
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred", ## guess label by directory structure
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2, ## validation image 20%
        subset="training",
        seed=123
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    print("클래스 : ", train_dataset.class_names)
    print("validation 클래스 : ", validation_dataset.class_names)
    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(28,28,3)
        ),
        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        ## find simple feature
        tf.keras.layers.Conv2D(
            32,
            (3,3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            (2,2)
        ),

        ## find detail feature
        tf.keras.layers.Conv2D(
            64,
            (3,3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            (2,2)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])


    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)

    print()
    print("======학습완료========")


if __name__ == "__main__":
    main()