import tensorflow as tf

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)


def load_dataset(
    directory,
    shuffle=True
):

    print(
        f"\nLoading dataset: {directory}"
    )

    dataset = (
        tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=shuffle,
            seed=SEED
        )
    )

    return dataset


def get_datasets():

    train_dataset = load_dataset(
        TRAIN_DIR,
        shuffle=True
    )

    val_dataset = load_dataset(
        VAL_DIR,
        shuffle=False
    )

    test_dataset = load_dataset(
        TEST_DIR,
        shuffle=False
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )