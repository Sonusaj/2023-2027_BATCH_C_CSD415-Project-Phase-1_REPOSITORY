import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

from msag import MSAG
from config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    LEARNING_RATE
)


def build_model():

    # ========================================================
    # INPUT
    # ========================================================

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        name="input_image"
    )

    # ========================================================
    # DATA AUGMENTATION
    # ========================================================

    x = layers.RandomFlip(
        "horizontal",
        name="random_flip"
    )(inputs)

    x = layers.RandomRotation(
        0.05,
        name="random_rotation"
    )(x)

    x = layers.RandomZoom(
        0.10,
        name="random_zoom"
    )(x)

    x = layers.RandomContrast(
        0.10,
        name="random_contrast"
    )(x)

    # ========================================================
    # EFFICIENTNETB0
    # ========================================================

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        name="efficientnetb0"
    )

    # Freeze backbone initially
    backbone.trainable = False

    x = backbone(
        x,
        training=False
    )

    # ========================================================
    # MSAG
    # ========================================================

    filters = int(
        x.shape[-1]
    )

    x = MSAG(
        filters=filters,
        name="msag"
    )(x)

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.4,
        name="dropout_1"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout_2"
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classification"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_MSAG"
    )

    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, backbone