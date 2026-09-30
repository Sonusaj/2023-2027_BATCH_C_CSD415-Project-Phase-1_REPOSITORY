import sys
import os
import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")   # save figures without opening a window
import matplotlib.pyplot as plt

from preprocessing import preprocess_image
from msag import MSAG
from config import MODEL_PATH, IMAGE_SIZE


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

# Automatically create results folder
os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# FIND EFFICIENTNET BACKBONE
# ============================================================

def find_efficientnet(model):

    # Check top-level layers
    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            name = layer.name.lower()

            if "efficientnet" in name:

                return layer

    # Search recursively
    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            try:

                result = find_efficientnet(layer)

                if result is not None:
                    return result

            except Exception:
                pass

    return None


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    # --------------------------------------------------------
    # First search nested EfficientNet
    # --------------------------------------------------------

    efficientnet = find_efficientnet(model)

    if efficientnet is not None:

        print(
            "EfficientNet backbone found:",
            efficientnet.name
        )

        for layer in reversed(
            efficientnet.layers
        ):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                print(
                    "Grad-CAM target layer:",
                    layer.name
                )

                return efficientnet, layer

    # --------------------------------------------------------
    # Search current model
    # --------------------------------------------------------

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                "Grad-CAM target layer:",
                layer.name
            )

            return model, layer

    # --------------------------------------------------------
    # Search nested models
    # --------------------------------------------------------

    for layer in reversed(
        model.layers
    ):

        if isinstance(
            layer,
            tf.keras.Model
        ):

            try:

                result = find_last_conv_layer(
                    layer
                )

                if result is not None:
                    return result

            except Exception:
                pass

    raise ValueError(
        "Could not find a convolutional layer "
        "for Grad-CAM."
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_gradcam_model():

    print("\nLoading model...")

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "MSAG": MSAG
        },
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Model name:",
        model.name
    )

    print(
        "Parameters:",
        model.count_params()
    )

    return model


# ============================================================
# CREATE GRAD-CAM
# ============================================================

def make_gradcam(image_path):

    print("\n========================================")
    print("GRAD-CAM DIABETIC RETINOPATHY")
    print("========================================")

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        print(
            "\nERROR: Image not found:"
        )

        print(image_path)

        return

    print(
        "\nInput image:"
    )

    print(image_path)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = load_gradcam_model()

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "MODEL LOADING ERROR"
        )

        print(
            "========================================"
        )

        print(e)

        return

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    print(
        "\nProcessing image..."
    )

    try:

        image = preprocess_image(
            image_path,
            IMAGE_SIZE
        )

    except Exception as e:

        print(
            "\nIMAGE PROCESSING ERROR"
        )

        print(e)

        return

    input_image = np.expand_dims(
        image,
        axis=0
    )

    print(
        "Processed image:",
        input_image.shape
    )

    # --------------------------------------------------------
    # Normal prediction first
    # --------------------------------------------------------

    print(
        "\nGenerating prediction..."
    )

    try:

        prediction = model.predict(
            input_image,
            verbose=0
        )[0]

    except Exception as e:

        print(
            "\nPREDICTION ERROR"
        )

        print(e)

        return

    predicted_class = int(
        np.argmax(
            prediction
        )
    )

    confidence = float(
        prediction[
            predicted_class
        ]
    )

    print(
        "\nPredicted class:",
        CLASS_NAMES[
            predicted_class
        ]
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    # ========================================================
    # FIND TARGET LAYER
    # ========================================================

    print(
        "\nSearching for Grad-CAM target layer..."
    )

    try:

        backbone, target_layer = (
            find_last_conv_layer(
                model
            )
        )

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "GRAD-CAM LAYER ERROR"
        )

        print(
            "========================================"
        )

        print(e)

        return

    print(
        "\nUsing backbone:",
        backbone.name
    )

    print(
        "Using layer:",
        target_layer.name
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    print(
        "\nGenerating Grad-CAM heatmap..."
    )

    try:

        # ----------------------------------------------------
        # Case 1:
        # EfficientNet backbone can be used directly
        # ----------------------------------------------------

        if backbone is not model:

            grad_model = tf.keras.models.Model(
                inputs=backbone.input,
                outputs=[
                    target_layer.output,
                    backbone.output
                ]
            )

            with tf.GradientTape() as tape:

                conv_outputs, backbone_output = (
                    grad_model(
                        input_image,
                        training=False
                    )
                )

                # The backbone output is then passed through
                # the remaining layers of the original model.
                x = backbone_output

                passed_backbone = False

                for layer in model.layers:

                    if layer is backbone:
                        passed_backbone = True
                        continue

                    if not passed_backbone:
                        continue

                    if isinstance(
                        layer,
                        tf.keras.layers.InputLayer
                    ):
                        continue

                    x = layer(
                        x,
                        training=False
                    )

                final_output = x

                class_score = (
                    final_output[
                        :,
                        predicted_class
                    ]
                )

            gradients = tape.gradient(
                class_score,
                conv_outputs
            )

        # ----------------------------------------------------
        # Case 2:
        # Target layer belongs directly to model
        # ----------------------------------------------------

        else:

            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[
                    target_layer.output,
                    model.output
                ]
            )

            with tf.GradientTape() as tape:

                conv_outputs, predictions = (
                    grad_model(
                        input_image,
                        training=False
                    )
                )

                class_score = (
                    predictions[
                        :,
                        predicted_class
                    ]
                )

            gradients = tape.gradient(
                class_score,
                conv_outputs
            )

        # ----------------------------------------------------
        # Check gradients
        # ----------------------------------------------------

        if gradients is None:

            raise RuntimeError(
                "Gradients are None. "
                "The selected layer is not connected "
                "correctly to the model output."
            )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs *
            pooled_gradients,
            axis=-1
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            max_value + 1e-8
        )

        heatmap = heatmap.numpy()

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "GRAD-CAM ERROR"
        )

        print(
            "========================================"
        )

        print(e)

        print(
            "\nThe normal prediction worked,"
            " but Grad-CAM could not be generated."
        )

        return


    # ========================================================
    # CREATE HEATMAP IMAGE
    # ========================================================

    heatmap = cv2.resize(
        heatmap,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    heatmap_uint8 = np.uint8(255 * heatmap)

    # OpenCV colormap is BGR -> convert to RGB for matplotlib
    heatmap_color = cv2.cvtColor(
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        ),
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # PREPROCESSED RETINA (what the model actually saw)
    # ========================================================

    preprocessed = np.clip(
        image, 0, 255
    ).astype(np.uint8)

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        preprocessed,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # ========================================================
    # OUTPUT FILENAMES (all inside results folder)
    # ========================================================

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    analysis_path = os.path.join(
        RESULT_DIR,
        image_name + "_gradcam_analysis.png"
    )

    # ========================================================
    # SAVE 3-PANEL FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    axes[0].imshow(preprocessed)
    axes[0].set_title("Preprocessed Retina")

    axes[1].imshow(heatmap_color)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"{CLASS_NAMES[predicted_class]}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        analysis_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # ALSO SAVE INDIVIDUAL IMAGES
    # ========================================================

    def save_rgb(name, rgb_image):

        path = os.path.join(
            RESULT_DIR,
            image_name + name
        )

        cv2.imwrite(
            path,
            cv2.cvtColor(
                rgb_image,
                cv2.COLOR_RGB2BGR
            )
        )

        return path

    save_rgb("_preprocessed.jpg", preprocessed)
    save_rgb("_heatmap.jpg", heatmap_color)
    save_rgb("_overlay.jpg", overlay)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("GRAD-CAM COMPLETED")
    print("========================================")

    print("\nPrediction :", CLASS_NAMES[predicted_class])
    print(f"Confidence : {confidence * 100:.2f}%")

    print("\nSaved in results folder:")
    print(RESULT_DIR)
    print(" -", os.path.basename(analysis_path))
    print(" -", image_name + "_preprocessed.jpg")
    print(" -", image_name + "_heatmap.jpg")
    print(" -", image_name + "_overlay.jpg")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print('python src\\gradcam.py "path\\to\\image.jpg"')

        sys.exit(1)

    make_gradcam(sys.argv[1])