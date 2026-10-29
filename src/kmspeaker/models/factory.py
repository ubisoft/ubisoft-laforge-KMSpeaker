# External Modules
from omegaconf import OmegaConf

# Internal Modules
from kmspeaker.models.transformer_adaln.model import TransformerKeypointAdaIN


def get_model(model_cfg, audio_encoder, num_keypoint):
    """Builds the model specified by model_cfg.type, given num_keypoint (from the dataset)."""
    # Ensure model_cfg is an OmegaConf object for consistent attribute access
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)

    input_dim = model_cfg.get("input_dim", model_cfg.pca_dim)
    if model_cfg.type == "TransformerKeypointAdaIN":
        keypoint_dim = num_keypoint * 3  # Num of keypoints times 3 for x, y, z coordinates
        return TransformerKeypointAdaIN(
            audio_encoder=audio_encoder,
            input_dim=input_dim,
            keypoint_dim=keypoint_dim,
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 8),
            mlp_ratio=model_cfg.get("mlp_ratio", 4),
            align_mask_width=model_cfg.get("align_mask_width", 1),
            feature_dim=model_cfg["embed_dim"],
            sequence_length=model_cfg["sequence_length"],
            use_learnable_pe=model_cfg.get("use_learnable_pe", True),
            use_keypoint_style_encoder=model_cfg.get("use_keypoint_style_encoder", False),
        )

    else:
        raise ValueError(f"Unsupported model type: {model_cfg.type}")
