import torch
import torch.nn as nn


class MSAG(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden_channels = max(
            channels // reduction,
            32
        )

        # ====================================================
        # CHANNEL ATTENTION
        # ====================================================

        self.channel_attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.Sigmoid()
        )

        # ====================================================
        # SPATIAL ATTENTION
        # ====================================================

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(
                2,
                1,
                kernel_size=7,
                padding=3,
                bias=False
            ),

            nn.Sigmoid()
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # Channel attention
        channel_map = self.channel_attention(x)

        x_channel = x * channel_map

        # Spatial information
        average_map = torch.mean(
            x_channel,
            dim=1,
            keepdim=True
        )

        maximum_map, _ = torch.max(
            x_channel,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [
                average_map,
                maximum_map
            ],
            dim=1
        )

        # Spatial attention
        spatial_map = self.spatial_attention(
            spatial_input
        )

        x_spatial = (
            x_channel
            * spatial_map
        )

        # Residual connection
        return x + x_spatial