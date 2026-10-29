# External Modules
import torch
import torch.nn as nn

# Internal Modules
from kmspeaker.models.diffpose.common import PositionalEncoding

### Style encode adapted from https://github.com/ubisoft/ubisoft-laforge-msmd/blob/main/style_encoder.py
class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims  # Tuple of dimensions to permute

    def forward(self, x):
        return x.permute(*self.dims)


class StyleEncoder(nn.Module):
    def __init__(self, style_encoder_cfg, keypoint_input_dim) -> None:
        super().__init__()

        self.input_dim = keypoint_input_dim

        self.motion_coef_dim = self.input_dim
        self.conv_feature_dim = style_encoder_cfg.conv_feature_dim
        self.output_size = style_encoder_cfg.d_style * 2

        self.pre_conv_permute = Permute((0, 2, 1))
        self.post_conv_permute = Permute((0, 2, 1))
        # these are the input layers
        self.input_layers = [
            # conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.motion_coef_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.conv_feature_dim, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.2),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.conv_feature_dim),
        ]
        self.input_layers = nn.Sequential(*self.input_layers)

        # apply positional encoding
        self.PE = PositionalEncoding(self.conv_feature_dim)

        # one transformer decoder
        self.encoder = nn.TransformerEncoderLayer(d_model=self.conv_feature_dim, nhead=8, dim_feedforward=self.conv_feature_dim, activation="gelu", batch_first=True)

        # end up with two more 1D conv layers
        self.output_layers = [
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.conv_feature_dim, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
            nn.Dropout(0.1),
            nn.ELU(),
            # apply layer norm
            nn.LayerNorm(self.output_size),
            # second conv1d
            self.pre_conv_permute,
            nn.Conv1d(in_channels=self.output_size, out_channels=self.output_size, kernel_size=3, padding=1),
            self.post_conv_permute,
        ]
        self.output_layers = nn.Sequential(*self.output_layers)

    def forward(self, motion_coef, do_sample=False):
        # Motion
        motion_feat = self.input_layers(motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder(motion_feat)

        out = self.output_layers(feat)

        # average pooling
        out = out.mean(dim=1)

        mu = out[:, : self.output_size // 2]
        logvar = out[:, self.output_size // 2 :]  # this cannot have a relu since we need negative values!!!!!

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        # determine if we should sample or not
        if do_sample:
            return mu + eps * std
        else:
            out = mu + eps * std
            return out, (mu, logvar)

    def sample(self, motion_coef):
        _, (mu, logvar) = self.forward(motion_coef)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
