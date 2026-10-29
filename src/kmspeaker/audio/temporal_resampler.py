# External Modules
import torch.nn as nn
import torch.nn.functional as F


class LinearResampler(nn.Module):
    """
    Adjusts the temporal resolution of features using linear interpolation.
    """

    def __init__(self, input_fps, output_fps):
        super().__init__()
        self.input_fps = input_fps
        self.output_fps = output_fps

        if self.input_fps <= 0 or self.output_fps <= 0:
            raise ValueError("Input and output FPS must be positive.")

    def forward(self, features, target_length=None):
        """
        Args:
            features (torch.Tensor): Combined features (N x T_current_fps x model_dim).

        Returns:
            torch.Tensor: Resampled features (N x T_animation_fps x model_dim).
        """
        N, T_current, D = features.shape

        if self.input_fps == self.output_fps:
            return features  # No resampling needed

        if T_current == 0:  # Handle empty sequences
            raise ValueError("Input features cannot have zero temporal dimension (T_current).")

        # T_output = T_input * (output_fps / input_fps)
        if target_length is None:
            target_length = int(T_current * (self.output_fps / self.input_fps))

        if target_length == T_current:  # If after calculation it's the same, no resampling
            return features

        if target_length == 0:  # Avoid creating empty tensor if original had content
            target_length = 1

        # Interpolate expects (N, D, T)
        features_permuted = features.permute(0, 2, 1)  # N x D x T_current

        resampled_features_permuted = F.interpolate(features_permuted, size=target_length, mode="linear", align_corners=False)

        # Permute back to (N, T_animation_fps, D)
        resampled_features = resampled_features_permuted.permute(0, 2, 1)

        return resampled_features
