import cv2
import numpy as np


def preprocess_image(
    image_path,
    image_size=224
):

    # Read image
    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # GREEN CHANNEL
    # ========================================================

    green = image[:, :, 1]

    # ========================================================
    # CLAHE
    # ========================================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    green_clahe = clahe.apply(
        green
    )

    # ========================================================
    # GRAYSCALE -> RGB
    # ========================================================

    processed = cv2.cvtColor(
        green_clahe,
        cv2.COLOR_GRAY2RGB
    )

    # ========================================================
    # RESIZE
    # ========================================================

    processed = cv2.resize(
        processed,
        (image_size, image_size)
    )

    # Float32
    processed = processed.astype(
        np.float32
    )

    return processed