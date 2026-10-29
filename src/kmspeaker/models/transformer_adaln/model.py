# External Modules
import torch
import torch.nn as nn

# Internal Modules
from kmspeaker.models.diffpose.common import PositionalEncoding
from kmspeaker.models.diffpose.common import enc_dec_mask

# Local Modules
from .components import FourierEncoder
from .components import TransformerDecoderAdaLN
from .components import TransformerDecoderLayerAdaLN


class TransformerKeypointAdaIN(nn.Module):
    def __init__(
        self,
        audio_encoder,
        input_dim,
        keypoint_dim,
        n_heads=8,
        n_layers=8,
        mlp_ratio=4,
        align_mask_width=1,
        feature_dim=256,
        sequence_length=100,
        use_learnable_pe=True,
        use_keypoint_style_encoder=False,
        device="cuda",
    ):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.motion_feat_dim = input_dim
        self.keypoint_dim = keypoint_dim
        self.use_keypoint_style_encoder = use_keypoint_style_encoder

        # Transformer parameters
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = use_learnable_pe
        # sequence length
        self.n_motions = sequence_length

        # Time embedding for the diffusion time step
        self.TE = FourierEncoder(dim=self.feature_dim)
        self.diff_step_map = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.GELU(), nn.Linear(self.feature_dim, self.feature_dim))

        # Combined conditioning dimension (time)
        self.conditioning_dim = self.feature_dim

        self.keypoint_proj = nn.Sequential(nn.Linear(self.keypoint_dim, self.feature_dim), nn.GELU(), nn.Linear(self.feature_dim, self.feature_dim))
        self.conditioning_dim += self.feature_dim  # time_dim + proj keypoint_dim

        if self.use_keypoint_style_encoder:
            self.conditioning_dim += self.feature_dim  # add style feature dimension

        # Conditioning projection to unified dimension
        self.conditioning_proj = nn.Sequential(nn.Linear(self.conditioning_dim, self.feature_dim), nn.GELU(), nn.Linear(self.feature_dim, self.feature_dim))

        # Positional encoding - only for motion tokens (no conditioning tokens prepended)
        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(1, self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # Motion feature projection
        self.feature_proj = nn.Linear(self.motion_feat_dim, self.feature_dim)

        # Transformer decoder with AdaLN
        decoder_layer = TransformerDecoderLayerAdaLN(
            d_model=self.feature_dim,
            nhead=self.n_heads,
            dim_feedforward=self.mlp_ratio * self.feature_dim,
            conditioning_dim=self.feature_dim,  # Unified conditioning dimension
            dropout=0.1,
            activation="gelu",
            sequential_condition=True,
        )

        self.transformer = TransformerDecoderAdaLN(decoder_layer, num_layers=self.n_layers)

        # Alignment mask for cross-attention (no conditioning tokens, so standard mask)
        if self.align_mask_width > 0:
            motion_len = self.n_motions
            alignment_mask = enc_dec_mask(motion_len, motion_len, 1, self.align_mask_width - 1)
            self.register_buffer("alignment_mask", alignment_mask)
        else:
            self.alignment_mask = None

        # Motion decoder
        self.motion_dec = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim // 2), nn.GELU(), nn.Linear(self.feature_dim // 2, self.motion_feat_dim))

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, t, conditioning, return_audio_features=False):
        """
        Forward pass of the TransformerKeypointAdaIN.

        Args:
            x (torch.Tensor): Noisy motion features of shape (N, L, d_motion).
            t (torch.Tensor): Diffusion time steps of shape (N,).
            conditioning (dict): Dictionary containing 'audio', 'keypoint', and optionally 'keypoint_style'.
            return_audio_features (bool): Whether to return audio features.

        Returns:
            torch.Tensor: Predicted denoised motion features of shape (N, L, d_motion).
        """
        # Extract conditioning
        audio = conditioning["audio"]
        keypoint = conditioning.get("keypoint", None)  # Temporal keypoint signal
        keypoint_style = conditioning.get("keypoint_style", None)  # Precomputed style feature

        # Encode audio
        audio = self.audio_encoder(audio)

        # Expand time embedding to match sequence length
        time_embedding = self.diff_step_map(self.TE(t))  # (N, feature_dim)
        time_embedding = time_embedding.unsqueeze(1).expand(-1, x.size(1), -1)  # (N, seq_len, feature_dim)
        ### copy to batch size, currently is (1, seq_len, feature_dim), must be (batch, seq_len, feature_dim)
        time_embedding = time_embedding.expand(x.size(0), -1, -1)  # (N, seq_len, feature_dim)

        combined_conditioning = [time_embedding]

        # Keypoint embedding (temporal)
        keypoint_embedding = self.keypoint_proj(keypoint)  # (N, seq_len, feature_dim)
        # Combine time, keypoint embedding, and keypoint style conditioning
        combined_conditioning.append(keypoint_embedding)
        if self.use_keypoint_style_encoder:
            assert keypoint_style is not None, "Keypoint style feature must be provided when using style encoder."
            combined_conditioning.append(keypoint_style.unsqueeze(1).expand(-1, keypoint.size(1), -1))  # Expand style feature

        combined_conditioning = torch.cat(combined_conditioning, dim=-1)  # (N, seq_len, conditioning_dim)
        unified_conditioning = self.conditioning_proj(combined_conditioning)  # (N, seq_len, feature_dim)

        # Project motion features
        feats_in = self.feature_proj(x)  # (N, L, feature_dim)

        # Add positional encoding (no conditioning tokens prepended in AdaLN version)
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # Transformer processing with AdaLN conditioning
        feat_out = self.transformer(tgt=feats_in, memory=audio, conditioning=unified_conditioning, memory_mask=self.alignment_mask)

        # Decode motion features (no conditioning tokens to skip)
        motion_feat_target = self.motion_dec(feat_out)  # (N, L, d_motion)

        if return_audio_features:
            return motion_feat_target, audio
        else:
            return motion_feat_target, None
