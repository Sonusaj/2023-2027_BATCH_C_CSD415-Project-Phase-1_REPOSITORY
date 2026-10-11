import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b1,
    EfficientNet_B1_Weights
)

from msag import MSAG

import config


class EfficientNetB1MSAG(nn.Module):

    def __init__(
        self,
        num_classes=config.NUM_CLASSES
    ):

        super().__init__()

        # ====================================================
        # EFFICIENTNET-B1
        # ====================================================

        weights = EfficientNet_B1_Weights.DEFAULT

        self.backbone_model = efficientnet_b1(
            weights=weights
        )

        # Keep this structure for V16 compatibility
        self.features = (
            self.backbone_model.features
        )

        self.filters = 1280

        # ====================================================
        # FREEZE BACKBONE
        # ====================================================

        for parameter in self.features.parameters():

            parameter.requires_grad = False

        # ====================================================
        # MSAG
        # ====================================================

        self.msag = MSAG(
            channels=self.filters
        )

        # ====================================================
        # GLOBAL POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                self.filters,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.Dropout(
                config.DROPOUT_1
            ),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                config.DROPOUT_2
            ),

            nn.Linear(
                128,
                num_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.features(x)

        x = self.msag(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x

    # ========================================================
    # UNFREEZE LAST BLOCKS
    # ========================================================

    def unfreeze_last_blocks(
        self,
        number_of_blocks=2
    ):

        for parameter in self.features.parameters():

            parameter.requires_grad = False

        total_blocks = len(
            self.features
        )

        start_index = max(
            0,
            total_blocks - number_of_blocks
        )

        for block in self.features[start_index:]:

            for parameter in block.parameters():

                parameter.requires_grad = True

    # ========================================================
    # FREEZE BACKBONE BATCH NORMALIZATION
    # ========================================================

    def freeze_backbone_batchnorm(self):

        for module in self.features.modules():

            if isinstance(
                module,
                nn.BatchNorm2d
            ):

                module.eval()

                for parameter in module.parameters():

                    parameter.requires_grad = False

    # ========================================================
    # PARAMETER INFORMATION
    # ========================================================

    def print_trainable_parameters(self):

        total = 0
        trainable = 0

        for parameter in self.parameters():

            total += parameter.numel()

            if parameter.requires_grad:

                trainable += parameter.numel()

        print(
            f"\nTotal parameters     : {total:,}"
        )

        print(
            f"Trainable parameters : {trainable:,}"
        )

        print(
            f"Frozen parameters    : "
            f"{total - trainable:,}"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = EfficientNetB1MSAG()

    model.to(
        config.DEVICE
    )

    model.print_trainable_parameters()

    x = torch.randn(
        2,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ).to(
        config.DEVICE
    )

    with torch.no_grad():

        output = model(x)

    print(
        "\nOutput shape:",
        output.shape
    )