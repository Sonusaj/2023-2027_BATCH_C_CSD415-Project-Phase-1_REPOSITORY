import os


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diabetic_retinopathy_model.keras"
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "training_checkpoint.keras"
)


# ============================================================
# DATASET
# ============================================================

# ============================================================
# DATASET
# ============================================================

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

# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

SEED = 42


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 15

LEARNING_RATE = 1e-4