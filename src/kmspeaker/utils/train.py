# Standard Library
import logging
from pathlib import Path

# External Modules
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW

logger = logging.getLogger(__name__)


def initialize_optional_models(cfg, device, num_keypoint, num_extra_style_keypoint):
    """Initialize optional encoder models based on trainer configuration flags."""

    keypoint_style_encoder = None

    if cfg.trainer.use_keypoint_style_encoder:
        # Internal Modules
        from kmspeaker.models.style_encoder.keypoint_style_encoder import StyleEncoder

        logger.info("Initializing keypoint style encoder...")
        keypoint_input_dim = num_keypoint * 3
        if cfg.model.keypoint_style_encoder.use_extra_style_keypoints:
            keypoint_input_dim += num_extra_style_keypoint * 3
        keypoint_style_encoder = StyleEncoder(cfg.model.keypoint_style_encoder, keypoint_input_dim).to(device)
    return keypoint_style_encoder


def resume_from_checkpoint(cfg, base_logdir, device, model, keypoint_style_encoder=None):
    """Load model and optional encoder weights from a checkpoint if cfg.model.checkpoint is set."""
    if not cfg.model.checkpoint:
        return

    checkpoint_path = base_logdir / Path(cfg.model.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Resuming model weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    optional_encoders = [
        (keypoint_style_encoder, "keypoint_style_encoder_state_dict", "keypoint style encoder"),
    ]
    for encoder, key, name in optional_encoders:
        if encoder is not None and key in checkpoint:
            logger.info(f"Loading {name} weights from checkpoint.")
            encoder.load_state_dict(checkpoint[key])


def log_model_parameter_counts(model, keypoint_style_encoder=None):
    """Log the number of trainable parameters for the main model and optional encoders."""

    def count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    logger.info(f"Model parameters: {count(model)}")
    if keypoint_style_encoder is not None:
        logger.info(f"Keypoint style encoder parameters: {count(keypoint_style_encoder)}")


def build_optimizer(cfg, model):
    """Instantiate and return the optimizer based on cfg.optimizer.type."""
    if cfg.optimizer.type != "AdamW":
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer.type}")

    logger.info(f"Initializing optimizer: {cfg.optimizer.type}")
    return get_AdamW_optimizer(model, cfg.optimizer)


def build_lr_scheduler(cfg, optimizer):
    """Instantiate and return the LR scheduler based on cfg.lr_scheduler.type, or None."""
    # External Modules
    from torch.optim.lr_scheduler import CosineAnnealingLR

    logger.info(f"Initializing LR scheduler: {cfg.lr_scheduler.type}")
    if cfg.lr_scheduler.type == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.trainer.total_training_steps,
            eta_min=cfg.lr_scheduler.eta_min,
        )
    logger.info("No LR scheduler or unsupported type specified.")
    return None


def build_checkpoint_data(cfg, global_step, model, optimizer, lr_scheduler, keypoint_metadata=None, keypoint_style_encoder=None):
    """Build a checkpoint dictionary with model and optional encoder state dicts."""
    checkpoint_data = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "keypoints": keypoint_metadata,
    }
    optional_encoders = [
        (keypoint_style_encoder, "keypoint_style_encoder_state_dict"),
    ]
    for encoder, key in optional_encoders:
        if encoder is not None:
            checkpoint_data[key] = encoder.state_dict()
    return checkpoint_data


def get_AdamW_optimizer(model, cfg_optimizer):
    # Support model as a list of modules
    if isinstance(model, list):
        params = []
        for m in model:
            # Only add parameters that require grad (skip frozen modules)
            params += [p for p in m.parameters() if p.requires_grad]
        return AdamW(params, lr=cfg_optimizer.lr, weight_decay=cfg_optimizer.weight_decay)
    return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg_optimizer.lr, weight_decay=cfg_optimizer.weight_decay)
