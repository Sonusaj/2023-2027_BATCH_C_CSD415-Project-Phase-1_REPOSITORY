import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import get_datasets
from config import (
    MODEL_PATH,
    CLASS_NAMES
)

from msag import MSAG


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


# ============================================================
# DATASET
# ============================================================

print("\nLoading test dataset...")

_, _, test_dataset = get_datasets()

print("\nDatasets loaded successfully.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "MSAG": MSAG
    },
    compile=True
)

print("\nModel loaded successfully.")

print("Model:", model.name)

print(
    "Parameters:",
    model.count_params()
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating on test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        np.argmax(
            predictions,
            axis=1
        )
    )


y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TEST RESULTS")
print("========================================")

print(
    f"Test Loss       : {test_loss:.4f}"
)

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Precision       : {precision * 100:.2f}%"
)

print(
    f"Recall          : {recall * 100:.2f}%"
)

print(
    f"F1 Score        : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")