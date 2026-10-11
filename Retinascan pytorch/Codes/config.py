import os
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "selected_images"
)


TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)


VAL_DIR = os.path.join(
    DATASET_DIR,
    "val"
)


TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GPU SETTINGS
# ============================================================

USE_AMP = torch.cuda.is_available()

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 256

CHANNELS = 3


# ============================================================
# CLASSES
# ============================================================

NUM_CLASSES = 5


CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# DATASET SETTINGS
# ============================================================

IMAGES_PER_CLASS = 911


# ============================================================
# ORIGINAL B1 TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 15

STAGE_1_EPOCHS = 5

STAGE_2_EPOCHS = 10


HEAD_LEARNING_RATE = 0.0002

FINE_TUNE_LEARNING_RATE = 0.00002

WEIGHT_DECAY = 0.00005


# ============================================================
# DROPOUT
# ============================================================

DROPOUT_1 = 0.20

DROPOUT_2 = 0.10


# ============================================================
# LABEL SMOOTHING
# ============================================================

LABEL_SMOOTHING = 0.0


# ============================================================
# ORIGINAL FINE-TUNING SETTINGS
# ============================================================

FINE_TUNE_LAST_BLOCKS = 3


# ============================================================
# IMPROVED B1 MODEL PATHS
# ============================================================

IMPROVED_B1_BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_improved_best.pth"
)


IMPROVED_B1_FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_improved_final.pth"
)


# ============================================================
# V3 MODEL PATHS
# ============================================================

V3_START_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_improved_best.pth"
)


V3_BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v3_best.pth"
)


V3_FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v3_final.pth"
)


V3_CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v3_checkpoint.pth"
)


V3_HEAD_EPOCHS = 2

V3_FINE_TUNE_EPOCHS = 10

V3_FINE_TUNE_LAST_BLOCKS = 2

V3_HEAD_LEARNING_RATE = 0.000003

V3_BACKBONE_LEARNING_RATE = 0.0000005

V3_WEIGHT_DECAY = 0.00005

V3_FOCAL_GAMMA = 2.0

V3_GRADIENT_CLIP_NORM = 1.0

V3_EARLY_STOPPING_PATIENCE = 3

V3_USE_AMP = False


# ============================================================
# V4 IMPROVEMENT SETTINGS
# ============================================================

# V4 starts from the already trained Improved B1 model.
# V3 remains completely untouched.

V4_START_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_improved_best.pth"
)


# ------------------------------------------------------------
# V4 Stage 1
# ------------------------------------------------------------

V4_HEAD_EPOCHS = 3


# ------------------------------------------------------------
# V4 Stage 2
# ------------------------------------------------------------

V4_FINE_TUNE_EPOCHS = 12


# ------------------------------------------------------------
# V4 fine-tuning
# ------------------------------------------------------------

V4_FINE_TUNE_LAST_BLOCKS = 3


# ------------------------------------------------------------
# V4 learning rates
# ------------------------------------------------------------

V4_HEAD_LEARNING_RATE = 0.00002

V4_BACKBONE_LEARNING_RATE = 0.000005


# ------------------------------------------------------------
# V4 regularization
# ------------------------------------------------------------

V4_WEIGHT_DECAY = 0.0001


# ------------------------------------------------------------
# V4 Moderate DR weighting
# ------------------------------------------------------------

V4_CLASS_WEIGHTS = [
    1.0,    # No DR
    1.0,    # Mild DR
    1.5,    # Moderate DR
    1.0,    # Severe DR
    1.0     # Proliferative DR
]


# ------------------------------------------------------------
# V4 gradient clipping
# ------------------------------------------------------------

V4_GRADIENT_CLIP_NORM = 1.0


# ------------------------------------------------------------
# V4 early stopping
# ------------------------------------------------------------

V4_EARLY_STOPPING_PATIENCE = 4


# ------------------------------------------------------------
# V4 AMP
# ------------------------------------------------------------

V4_USE_AMP = torch.cuda.is_available()


# ============================================================
# V4 MODEL PATHS
# ============================================================

V4_BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v4_best.pth"
)


V4_FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v4_final.pth"
)


V4_CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_v4_checkpoint.pth"
)


# ============================================================
# ORIGINAL B1 MODEL PATHS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_best.pth"
)


FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_final.pth"
)


CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "retina_scan_efficientnet_b1_checkpoint.pth"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_config():

    print("=" * 70)

    print(
        "RETINA-SCAN - EFFICIENTNET-B1 V4"
    )

    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


        gpu_memory = (

            torch.cuda
            .get_device_properties(0)
            .total_memory
            /
            (1024 ** 3)
        )


        print(
            f"GPU Memory: "
            f"{gpu_memory:.2f} GB"
        )

    else:

        print(
            "GPU: Not available"
        )

        print(
            "WARNING: Training will use CPU."
        )


    print(
        "Image Size:",
        IMAGE_SIZE
    )


    print(
        "Batch Size:",
        BATCH_SIZE
    )


    print(
        "Workers:",
        NUM_WORKERS
    )


    print(
        "V4 AMP:",
        V4_USE_AMP
    )


    print(
        "Classes:",
        NUM_CLASSES
    )


    print(
        "Images per class:",
        IMAGES_PER_CLASS
    )


    print(
        "V4 Stage 1 epochs:",
        V4_HEAD_EPOCHS
    )


    print(
        "V4 Stage 2 epochs:",
        V4_FINE_TUNE_EPOCHS
    )


    print(
        "V4 unfrozen blocks:",
        V4_FINE_TUNE_LAST_BLOCKS
    )


    print(
        "V4 head learning rate:",
        V4_HEAD_LEARNING_RATE
    )


    print(
        "V4 backbone learning rate:",
        V4_BACKBONE_LEARNING_RATE
    )


    print(
        "V4 weight decay:",
        V4_WEIGHT_DECAY
    )


    print(
        "V4 class weights:",
        V4_CLASS_WEIGHTS
    )


    print(
        "V4 gradient clip:",
        V4_GRADIENT_CLIP_NORM
    )


    print(
        "V4 early stopping:",
        V4_EARLY_STOPPING_PATIENCE
    )


    print()

    print(
        "V4 starting model:"
    )

    print(
        V4_START_MODEL_PATH
    )


    print()

    print(
        "V4 best model:"
    )

    print(
        V4_BEST_MODEL_PATH
    )


    print()

    print(
        "V4 final model:"
    )

    print(
        V4_FINAL_MODEL_PATH
    )


    print("=" * 70)

