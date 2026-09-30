import os
import tensorflow as tf

from dataset import get_datasets
from model import build_model
from config import MODEL_PATH, EPOCHS, MODEL_DIR
from msag import MSAG


print("\n========================================")
print("DIABETIC RETINOPATHY MODEL TRAINING")
print("========================================")


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "training_checkpoint.keras"
)

EPOCH_FILE = os.path.join(
    MODEL_DIR,
    "last_epoch.txt"
)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset, val_dataset, test_dataset = get_datasets()

print("\nDatasets loaded successfully.")


# ============================================================
# LOAD CHECKPOINT OR CREATE MODEL
# ============================================================

if os.path.exists(CHECKPOINT_PATH):

    print("\n========================================")
    print("CHECKPOINT FOUND")
    print("========================================")

    print("\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    try:

        model = tf.keras.models.load_model(
            CHECKPOINT_PATH,
            custom_objects={
                "MSAG": MSAG
            },
            compile=True
        )

    except Exception as e:

        print("\nERROR loading checkpoint:")
        print(e)

        print("\nTrying to load without compilation...")

        model = tf.keras.models.load_model(
            CHECKPOINT_PATH,
            custom_objects={
                "MSAG": MSAG
            },
            compile=False
        )

        # Recompile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    # --------------------------------------------------------
    # Read completed epoch
    # --------------------------------------------------------

    if os.path.exists(EPOCH_FILE):

        with open(
            EPOCH_FILE,
            "r"
        ) as f:

            initial_epoch = int(
                f.read().strip()
            )

    else:

        initial_epoch = 0


else:

    print("\n========================================")
    print("NO CHECKPOINT FOUND")
    print("========================================")

    print("\nCreating new model...")

    model, backbone = build_model()

    initial_epoch = 0


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n========================================")
print("MODEL INFORMATION")
print("========================================")

print("Model name :", model.name)
print("Parameters :", model.count_params())

print("\nStarting from epoch:", initial_epoch + 1)
print("Training until epoch:", EPOCHS)


# ============================================================
# CHECK IF TRAINING IS ALREADY COMPLETE
# ============================================================

if initial_epoch >= EPOCHS:

    print("\n========================================")
    print("TRAINING ALREADY COMPLETED")
    print("========================================")

    print(
        f"Checkpoint is already at epoch "
        f"{initial_epoch}."
    )

    print(
        f"Configured total epochs: {EPOCHS}"
    )

    print("\nIncrease EPOCHS in config.py if")
    print("you want to continue training.")

    exit()


# ============================================================
# EPOCH SAVER
# ============================================================

class EpochSaver(
    tf.keras.callbacks.Callback
):

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        completed_epoch = epoch + 1

        with open(
            EPOCH_FILE,
            "w"
        ) as f:

            f.write(
                str(completed_epoch)
            )

        print(
            f"\nEpoch {completed_epoch} completed."
        )

        print(
            "Resume information saved."
        )


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Save latest checkpoint
    tf.keras.callbacks.ModelCheckpoint(

        filepath=CHECKPOINT_PATH,

        save_best_only=False,

        save_weights_only=False,

        verbose=1
    ),

    # Save best final model
    tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        save_weights_only=False,

        verbose=1
    ),

    # Save epoch number
    EpochSaver(),

    # Reduce learning rate
    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7,

        verbose=1
    ),

    # Stop if validation accuracy stops improving
    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=5,

        restore_best_weights=True,

        verbose=1
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING / RESUMING TRAINING")
print("========================================")

print(
    f"\nEpoch {initial_epoch + 1} "
    f"→ Epoch {EPOCHS}"
)

print("\n")


history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    initial_epoch=initial_epoch,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

model.save(
    MODEL_PATH
)


# ============================================================
# COMPLETED
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print("\nBest model:")
print(MODEL_PATH)

print("\nCheckpoint:")
print(CHECKPOINT_PATH)

print("\nLast completed epoch:")

if os.path.exists(EPOCH_FILE):

    with open(
        EPOCH_FILE,
        "r"
    ) as f:

        print(
            f.read().strip()
        )

print("\n========================================")