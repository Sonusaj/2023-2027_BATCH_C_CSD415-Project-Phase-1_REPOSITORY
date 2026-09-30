import tensorflow as tf
from tensorflow.keras import layers


class MSAG(layers.Layer):

    def __init__(
        self,
        filters,
        reduction=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.reduction = reduction

        reduced_filters = max(
            filters // reduction,
            8
        )

        # Channel attention
        self.global_pool = layers.GlobalAveragePooling2D()

        self.channel_dense1 = layers.Dense(
            reduced_filters,
            activation="relu"
        )

        self.channel_dense2 = layers.Dense(
            filters,
            activation="sigmoid"
        )

        # Multi-scale spatial branches
        self.conv3 = layers.Conv2D(
            1,
            kernel_size=3,
            padding="same",
            activation="sigmoid"
        )

        self.conv5 = layers.Conv2D(
            1,
            kernel_size=5,
            padding="same",
            activation="sigmoid"
        )

        self.conv7 = layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid"
        )

        self.output_conv = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same"
        )

    def call(self, inputs):

        # Channel attention
        channel = self.global_pool(inputs)

        channel = self.channel_dense1(channel)

        channel = self.channel_dense2(channel)

        channel = tf.reshape(
            channel,
            [-1, 1, 1, self.filters]
        )

        channel_attention = inputs * channel

        # Multi-scale spatial attention
        attention3 = self.conv3(
            channel_attention
        )

        attention5 = self.conv5(
            channel_attention
        )

        attention7 = self.conv7(
            channel_attention
        )

        spatial_attention = (
            attention3 +
            attention5 +
            attention7
        ) / 3.0

        output = (
            channel_attention *
            spatial_attention
        )

        output = self.output_conv(output)

        # Residual connection
        output = output + inputs

        return output

    def get_config(self):

        config = super().get_config()

        config.update({
            "filters": self.filters,
            "reduction": self.reduction
        })

        return config