
import sys
import os

import numpy as np
import tensorflow as tf

from preprocessing import preprocess_image
from config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES
)

from msag import MSAG


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FINAL_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "diabetic_retinopathy_model.keras"
)

CHECKPOINT_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "training_checkpoint.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Try final model first
    # --------------------------------------------------------

    model_paths = [
        FINAL_MODEL,
        CHECKPOINT_MODEL
    ]

    for path in model_paths:

        print("\nTrying model:")
        print(path)

        if not os.path.exists(path):

            print("File not found.")

            continue

        try:

            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "MSAG": MSAG
                },
                compile=False
            )

            print("\nModel loaded successfully.")

            print("Model:", model.name)
            print("Parameters:", model.count_params())
            print("Input:", model.input_shape)
            print("Output:", model.output_shape)

            print("\nUsing:")
            print(path)

            return model

        except Exception as e:

            print("\nCould not load this model.")

            print(type(e).__name__)
            print(e)

    print("\n========================================")
    print("MODEL LOADING FAILED")
    print("========================================")

    print("\nNeither model could be loaded.")

    print("\nCheck:")
    print("1. The .keras file is complete.")
    print("2. src/msag.py exists.")
    print("3. TensorFlow/Keras versions are compatible.")

    return None


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image_path):

    print("\n========================================")
    print("DIABETIC RETINOPATHY PREDICTION")
    print("========================================")

    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if not os.path.exists(image_path):

        print("\nERROR: Image not found.")
        print(image_path)

        return

    print("\nImage:")
    print(image_path)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    model = load_model()

    if model is None:

        return

    # ========================================================
    # PREPROCESS
    # ========================================================

    print("\nProcessing image...")

    try:

        image = preprocess_image(
            image_path,
            IMAGE_SIZE
        )

    except Exception as e:

        print("\n========================================")
        print("IMAGE PROCESSING ERROR")
        print("========================================")

        print(e)

        return

    image = np.expand_dims(
        image,
        axis=0
    )

    print(
        "Processed image shape:",
        image.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print("\nGenerating prediction...")

    try:

        prediction = model.predict(
            image,
            verbose=0
        )[0]

    except Exception as e:

        print("\n========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(e)

        return

    # ========================================================
    # PREDICTED CLASS
    # ========================================================

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_class]
    )

    # ========================================================
    # RESULT
    # ========================================================

    print("\n========================================")
    print("PREDICTION RESULT")
    print("========================================")

    print(
        "\nPredicted class :",
        CLASS_NAMES[predicted_class]
    )

    print(
        "Class number    :",
        predicted_class
    )

    print(
        f"Confidence      : "
        f"{confidence * 100:.2f}%"
    )

    # ========================================================
    # ALL PROBABILITIES
    # ========================================================

    print("\nClass probabilities:")

    for i, probability in enumerate(
        prediction
    ):

        print(
            f"{CLASS_NAMES[i]:35s}"
            f": {probability * 100:6.2f}%"
        )

    print("\n========================================")
    print("PREDICTION COMPLETED")
    print("========================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python src\\predict.py image.jpg"
        )

        print("\nExample:")
        print(
            'python src\\predict.py '
            '"C:\\Users\\ABDUL\\Desktop\\retina.jpg"'
        )

        sys.exit(1)

    predict_image(
        sys.argv[1]
    )
