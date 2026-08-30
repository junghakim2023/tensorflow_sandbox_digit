import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "number_model.keras"
IMAGE_PATH = BASE_DIR / "data" / "test1.png"

def main():
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    image = tf.keras.utils.load_img(
        IMAGE_PATH,
        target_size = (28, 28)
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    image = tf.expand_dims(
        image,
        axis = 0
    )

    prediction = model.predict(
        image,
        verbose = 0
    )

    number = tf.argmax(
        prediction[0]
    ).numpy()

    confidence = prediction[0][number]

    print()
    print(f"예측 숫자 : {number}")
    print(f"확률 : {confidence: .2%}")
    print()

if __name__ == "__main__":
    main()