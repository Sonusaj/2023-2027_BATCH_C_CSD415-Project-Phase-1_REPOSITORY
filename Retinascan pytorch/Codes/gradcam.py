import os

import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.forward_hook_function
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.backward_hook_function
            )
        )

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook_function(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    # ========================================================
    # BACKWARD HOOK
    # ========================================================

    def backward_hook_function(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # ========================================================
    # GENERATE CAM
    # ========================================================

    def generate(
        self,
        image_tensor
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        score = output[
            0,
            predicted_class
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient not captured."
            )

        activations = (
            self.activations.detach()
        )

        gradients = (
            self.gradients.detach()
        )

        # Average gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1
        )

        cam = torch.relu(
            cam
        )

        cam = (
            cam[0]
            .cpu()
            .numpy()
        )

        # Normalize
        minimum = cam.min()
        maximum = cam.max()

        if maximum > minimum:

            cam = (
                cam - minimum
            ) / (
                maximum - minimum
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return (
            cam,
            predicted_class,
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    def remove(self):

        self.forward_hook.remove()

        self.backward_hook.remove()


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    cam,
    width,
    height
):

    cam = cv2.resize(
        cam,
        (width, height)
    )

    cam_uint8 = np.uint8(
        cam * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    return heatmap


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original,
    heatmap
):

    return cv2.addWeighted(
        original,
        0.60,
        heatmap,
        0.40,
        0
    )


# ============================================================
# COMBINED IMAGE
# ============================================================

def create_combined(
    original,
    heatmap,
    overlay
):

    original = cv2.resize(
        original,
        (400, 400)
    )

    heatmap = cv2.resize(
        heatmap,
        (400, 400)
    )

    overlay = cv2.resize(
        overlay,
        (400, 400)
    )

    combined = cv2.hconcat(
        [
            original,
            heatmap,
            overlay
        ]
    )

    return combined


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    original,
    cam,
    output_directory,
    base_name
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    height, width = original.shape[:2]

    heatmap = create_heatmap(
        cam,
        width,
        height
    )

    overlay = create_overlay(
        original,
        heatmap
    )

    combined = create_combined(
        original,
        heatmap,
        overlay
    )

    original_path = os.path.join(
        output_directory,
        base_name + "_original.jpg"
    )

    heatmap_path = os.path.join(
        output_directory,
        base_name + "_heatmap.jpg"
    )

    overlay_path = os.path.join(
        output_directory,
        base_name + "_gradcam.jpg"
    )

    combined_path = os.path.join(
        output_directory,
        base_name + "_combined.jpg"
    )

    cv2.imwrite(
        original_path,
        original
    )

    cv2.imwrite(
        heatmap_path,
        heatmap
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    cv2.imwrite(
        combined_path,
        combined
    )

    return {
        "original": original_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "combined": combined_path
    }