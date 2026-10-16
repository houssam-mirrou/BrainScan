
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = 224


def preprocess_image(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype: {gray.dtype}")

    height, width = gray.shape
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        gray,
        (new_width, new_height),
        interpolation=interpolation
    )

    canvas = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2

    canvas[y:y + new_height, x:x + new_width] = resized

    normalized = canvas.astype(np.float32) / 255.0

    return normalized[..., np.newaxis]


def create_dataset(records, training=False, batch_size=32):
    records = list(records)

    def generator():
        for path, label in records:
            image = preprocess_image(path)
            yield image, np.int32(CLASS_TO_ID[label])

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(224, 224, 1),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32
            )
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=512,
            seed=42,
            reshuffle_each_iteration=True
        )

    return dataset.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def build_model():
    layers = tf.keras.layers

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(4, activation="softmax")
    ])

    return model
