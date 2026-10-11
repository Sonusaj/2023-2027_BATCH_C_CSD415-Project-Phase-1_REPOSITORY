import os
import uuid
import traceback

import cv2
import numpy as np
import torch

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    send_from_directory,
    jsonify
)

from PIL import Image
from torchvision import transforms

import config
from model import EfficientNetB1MSAG


# ================================================================
# FLASK APPLICATION
# ================================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ================================================================
# BASE DIRECTORIES
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

GRADCAM_FOLDER = os.path.join(
    BASE_DIR,
    "gradcam"
)

TEMPLATE_FOLDER = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_FOLDER = os.path.join(
    BASE_DIR,
    "static"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    GRADCAM_FOLDER,
    exist_ok=True
)

os.makedirs(
    TEMPLATE_FOLDER,
    exist_ok=True
)

os.makedirs(
    STATIC_FOLDER,
    exist_ok=True
)


# ================================================================
# ALLOWED FILE TYPES
# ================================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


# ================================================================
# DEVICE
# ================================================================

DEVICE = config.DEVICE


# ================================================================
# CLASS NAMES
# ================================================================

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ================================================================
# IMAGE TRANSFORMATION
# ================================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ================================================================
# MODEL
# ================================================================

model = EfficientNetB1MSAG(
    num_classes=config.NUM_CLASSES
)


# ================================================================
# LATEST MODEL
# ================================================================

MODEL_PATH = os.path.join(
    config.MODEL_DIR,
    "retina_scan_efficientnet_b1_v16_best.pth"
)


if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        "\nLatest EfficientNet-B1 V16 model was not found.\n"
        f"Expected path:\n{MODEL_PATH}\n"
    )


print("=" * 70)
print("RETINA-SCAN FLASK APPLICATION")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

print(
    "Model:",
    MODEL_PATH
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# LOAD MODEL
# ================================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if isinstance(
    checkpoint,
    dict
):

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint
        )
    )

else:

    state_dict = checkpoint


# Remove DataParallel prefix if present

clean_state_dict = {

    key[7:] if key.startswith("module.") else key:
    value

    for key, value in state_dict.items()
}


model.load_state_dict(
    clean_state_dict,
    strict=True
)


model.to(
    DEVICE
)

model.eval()


print(
    "Model loaded successfully."
)

print("=" * 70)


# ================================================================
# ALLOWED FILE FUNCTION
# ================================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS

    )


# ================================================================
# FUNDUS IMAGE VALIDATION
# ================================================================

def looks_like_fundus(image):

    """
    Basic fundus-image validation.

    Checks:

    1. Dark outer corners
    2. Red-channel dominance
    3. Red-to-blue ratio
    4. Adequate central intensity
    """

    if (
        image is None
        or image.size == 0
    ):

        return False


    # ------------------------------------------------------------
    # Resize for fast validation
    # ------------------------------------------------------------

    small = cv2.resize(
        image,
        (256, 256)
    )


    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    ).astype(
        np.float32
    )


    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )


    # ------------------------------------------------------------
    # 1. DARK CORNER CHECK
    # ------------------------------------------------------------

    corners = np.concatenate([

        gray[
            :30,
            :30
        ].flatten(),

        gray[
            :30,
            -30:
        ].flatten(),

        gray[
            -30:,
            :30
        ].flatten(),

        gray[
            -30:,
            -30:
        ].flatten()

    ])


    corner_dark_ratio = np.mean(
        corners < 45
    )


    if corner_dark_ratio < 0.50:

        return False


    # ------------------------------------------------------------
    # 2. COLOR DOMINANCE CHECK
    # ------------------------------------------------------------

    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]


    mean_red = np.mean(
        red
    )

    mean_green = np.mean(
        green
    )

    mean_blue = np.mean(
        blue
    )


    if (
        mean_blue >= mean_red
        or
        (mean_green - mean_red) > 10
    ):

        return False


    # ------------------------------------------------------------
    # 3. RED / BLUE RATIO
    # ------------------------------------------------------------

    if (

        mean_blue > 0

        and

        (
            mean_red
            /
            (
                mean_blue
                + 1e-5
            )
        ) < 1.2

    ):

        return False


    # ------------------------------------------------------------
    # 4. CENTRAL INTENSITY
    # ------------------------------------------------------------

    center_region = gray[
        64:192,
        64:192
    ]


    center_mean = np.mean(
        center_region
    )


    if center_mean < 20:

        return False


    return True


# ================================================================
# UNIQUE FILE NAME
# ================================================================

def create_unique_name(
    original_filename
):

    extension = (

        original_filename
        .rsplit(
            ".",
            1
        )[1]
        .lower()

    )


    unique_id = uuid.uuid4().hex[:12]


    return (
        f"retina_{unique_id}.{extension}"
    )


# ================================================================
# GRAD-CAM
# ================================================================

def generate_gradcam(
    input_tensor,
    target_layer
):

    activation_holder = {
        "activation": None
    }


    # ------------------------------------------------------------
    # FORWARD HOOK
    # ------------------------------------------------------------

    def forward_hook(
        module,
        inputs,
        output
    ):

        activation_holder[
            "activation"
        ] = output


        if output.requires_grad:

            output.retain_grad()


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )


    try:

        model.zero_grad(
            set_to_none=True
        )


        image_tensor = (
            input_tensor
            .detach()
            .clone()
            .requires_grad_(True)
        )


        with torch.enable_grad():

            output = model(
                image_tensor
            )


            probabilities = (
                torch.softmax(
                    output,
                    dim=1
                )
            )


            predicted_class = int(
                torch.argmax(
                    probabilities,
                    dim=1
                ).item()
            )


            class_score = (
                output[
                    0,
                    predicted_class
                ]
            )


            class_score.backward()


        activation = (
            activation_holder[
                "activation"
            ]
        )


        if activation is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )


        gradient = activation.grad


        if gradient is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        activation = activation.detach()

        gradient = gradient.detach()


        # --------------------------------------------------------
        # GRADIENT WEIGHTS
        # --------------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )


        # --------------------------------------------------------
        # CLASS ACTIVATION MAP
        # --------------------------------------------------------

        cam = (
            weights
            *
            activation
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        cam = (
            cam[
                0
            ]
            .cpu()
            .numpy()
        )


        # --------------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------------

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )


        if (
            cam_max - cam_min
            > 1e-8
        ):

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam,
                dtype=np.float32
            )


        probabilities_np = (
            probabilities[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )


        return (
            cam,
            predicted_class,
            probabilities_np
        )


    finally:

        forward_handle.remove()


# ================================================================
# HEATMAP
# ================================================================

def create_heatmap(
    cam,
    width,
    height
):

    cam_resized = cv2.resize(
        cam,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )


    cam_uint8 = np.uint8(
        cam_resized * 255
    )


    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )


    return (
        cam_resized,
        heatmap
    )


# ================================================================
# OVERLAY
# ================================================================

def create_overlay(
    original_bgr,
    heatmap
):

    return cv2.addWeighted(
        original_bgr,
        0.60,
        heatmap,
        0.40,
        0
    )


# ================================================================
# COMBINED IMAGE
# ================================================================

def create_combined_image(
    original_bgr,
    heatmap,
    overlay
):

    height = 500


    def resize_height(
        image,
        target_height
    ):

        scale = (
            target_height
            /
            image.shape[0]
        )


        return cv2.resize(
            image,
            (
                int(
                    image.shape[1]
                    * scale
                ),
                target_height
            )
        )


    original_resized = resize_height(
        original_bgr,
        height
    )


    heatmap_resized = resize_height(
        heatmap,
        height
    )


    overlay_resized = resize_height(
        overlay,
        height
    )


    def add_label(
        image,
        text
    ):

        result = image.copy()


        cv2.rectangle(
            result,
            (0, 0),
            (
                result.shape[1],
                45
            ),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            result,
            text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        return result


    return cv2.hconcat([

        add_label(
            original_resized,
            "Original"
        ),

        add_label(
            heatmap_resized,
            "Grad-CAM Heatmap"
        ),

        add_label(
            overlay_resized,
            "Grad-CAM Overlay"
        )

    ])


# ================================================================
# HOME
# ================================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template(
        "index.html"
    )


# ================================================================
# PREDICTION
# ================================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ------------------------------------------------------------
    # FILE CHECK
    # ------------------------------------------------------------

    if (
        "file" not in request.files
        or
        request.files[
            "file"
        ].filename == ""
    ):

        return jsonify({
            "success": False,
            "error":
                "Please select a retinal fundus image."
        }), 400


    file = request.files[
        "file"
    ]


    # ------------------------------------------------------------
    # EXTENSION CHECK
    # ------------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return jsonify({
            "success": False,
            "error":
                "Unsupported file type. "
                "Upload JPG, JPEG, PNG, BMP, or WEBP."
        }), 400


    # ------------------------------------------------------------
    # SAVE IMAGE
    # ------------------------------------------------------------

    saved_filename = (
        create_unique_name(
            file.filename
        )
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        saved_filename
    )


    try:

        file.save(
            upload_path
        )

    except Exception:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error":
                "Failed to save the uploaded image."
        }), 500


    # ------------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------------

    original_bgr = cv2.imread(
        upload_path
    )


    if original_bgr is None:

        if os.path.exists(
            upload_path
        ):

            os.remove(
                upload_path
            )


        return jsonify({
            "success": False,
            "error":
                "The uploaded file could not be read as an image."
        }), 400


    # ------------------------------------------------------------
    # FUNDUS VALIDATION
    # ------------------------------------------------------------

    if not looks_like_fundus(
        original_bgr
    ):

        if os.path.exists(
            upload_path
        ):

            os.remove(
                upload_path
            )


        return jsonify({
            "success": False,
            "fundus_valid": False,
            "error":
                "Uploaded image is not a valid fundus image. "
                "Please upload a proper retinal fundus photograph."
        }), 400


    # ------------------------------------------------------------
    # CONVERT TO RGB
    # ------------------------------------------------------------

    original_rgb = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB
    )


    pil_image = Image.fromarray(
        original_rgb
    )


    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    input_tensor = (
        transform(
            pil_image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    # ------------------------------------------------------------
    # TARGET LAYER
    # ------------------------------------------------------------

    target_layer = (
        model.features[-1][0]
    )


    # ------------------------------------------------------------
    # PREDICTION + GRAD-CAM
    # ------------------------------------------------------------

    try:

        (
            cam,
            predicted_class,
            probabilities
        ) = generate_gradcam(
            input_tensor,
            target_layer
        )

    except Exception as error:

        traceback.print_exc()


        return jsonify({
            "success": False,
            "error":
                "Grad-CAM generation failed."
        }), 500


    # ------------------------------------------------------------
    # PREDICTION INFORMATION
    # ------------------------------------------------------------

    predicted_class_name = (
        CLASS_NAMES[
            predicted_class
        ]
    )


    confidence = float(
        probabilities[
            predicted_class
        ]
        * 100
    )


    probability_list = [

        {
            "stage": index,

            "name": CLASS_NAMES[index],

            "probability": round(
                float(
                    probabilities[index]
                ) * 100,
                2
            )

        }

        for index in range(
            config.NUM_CLASSES
        )

    ]


    # ------------------------------------------------------------
    # CREATE GRAD-CAM IMAGES
    # ------------------------------------------------------------

    height, width = (
        original_bgr.shape[:2]
    )


    _, heatmap = create_heatmap(
        cam,
        width,
        height
    )


    overlay = create_overlay(
        original_bgr,
        heatmap
    )


    combined = create_combined_image(
        original_bgr,
        heatmap,
        overlay
    )


    # ------------------------------------------------------------
    # FILE NAMES
    # ------------------------------------------------------------

    base_name = os.path.splitext(
        saved_filename
    )[0]


    heatmap_filename = (
        f"{base_name}_heatmap.jpg"
    )


    gradcam_filename = (
        f"{base_name}_gradcam.jpg"
    )


    combined_filename = (
        f"{base_name}_combined.jpg"
    )


    # ------------------------------------------------------------
    # SAVE GRAD-CAM RESULTS
    # ------------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            GRADCAM_FOLDER,
            heatmap_filename
        ),
        heatmap
    )


    cv2.imwrite(
        os.path.join(
            GRADCAM_FOLDER,
            gradcam_filename
        ),
        overlay
    )


    cv2.imwrite(
        os.path.join(
            GRADCAM_FOLDER,
            combined_filename
        ),
        combined
    )


    # ------------------------------------------------------------
    # RETURN RESULT
    # ------------------------------------------------------------

    return jsonify({

        "success": True,

        "fundus_valid": True,

        "prediction":
            predicted_class_name,

        "stage":
            predicted_class,

        "confidence":
            round(
                confidence,
                2
            ),

        "probabilities":
            probability_list,

        "original_image":
            url_for(
                "uploaded_file",
                filename=saved_filename
            ),

        "heatmap_image":
            url_for(
                "gradcam_file",
                filename=heatmap_filename
            ),

        "gradcam_image":
            url_for(
                "gradcam_file",
                filename=gradcam_filename
            ),

        "combined_image":
            url_for(
                "gradcam_file",
                filename=combined_filename
            )

    })


# ================================================================
# UPLOADED FILE ROUTE
# ================================================================

@app.route(
    "/uploads/<path:filename>",
    methods=["GET"]
)
def uploaded_file(
    filename
):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ================================================================
# GRAD-CAM FILE ROUTE
# ================================================================

@app.route(
    "/gradcam/<path:filename>",
    methods=["GET"]
)
def gradcam_file(
    filename
):

    return send_from_directory(
        GRADCAM_FOLDER,
        filename
    )


# ================================================================
# FILE TOO LARGE
# ================================================================

@app.errorhandler(413)
def file_too_large(
    error
):

    return jsonify({
        "success": False,
        "error":
            "File is too large. Maximum size is 10 MB."
    }), 413


# ================================================================
# INTERNAL ERROR
# ================================================================

@app.errorhandler(500)
def internal_server_error(
    error
):

    return jsonify({
        "success": False,
        "error":
            "An internal error occurred while processing the image."
    }), 500


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )