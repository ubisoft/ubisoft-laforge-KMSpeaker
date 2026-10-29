# Standard Library
import logging
from pathlib import Path

# External Modules
from omegaconf import OmegaConf

# Internal Modules
from kmspeaker.data.pca_cached_dataset import PCACachedDataSet

logger = logging.getLogger(__name__)


def get_data(data_cfg, device):
    """Builds the PCACachedDataSet described by data_cfg."""
    # Ensure data_cfg is an OmegaConf object for consistent attribute access
    if isinstance(data_cfg, dict):
        data_cfg = OmegaConf.create(data_cfg)

    data_root = Path(data_cfg.data_root)
    metadata_path = data_root / data_cfg.metadata_path
    animation_dataset_path = data_root / data_cfg.animation_dataset_path
    audio_pickle_path = data_root / data_cfg.audio_pickle_path
    pca_path = str(data_root / data_cfg.pca_path) if "pca_path" in data_cfg else None
    keypoint_index_pickle = data_root / data_cfg.keypoint_index_pickle
    template_data_path = str(data_root / data_cfg.template_path) if data_cfg.get("template_path", None) else None
    landmark_pickle_path = str(data_root / data_cfg.landmark_pickle_path)
    extra_style_processed_keypoint_path = (
        str(data_root / data_cfg.keypoint_style_encoder.extra_style_processed_keypoint_path) if data_cfg.keypoint_style_encoder.get("extra_style_processed_keypoint_path", None) else None
    )
    use_extra_style_keypoints = data_cfg.keypoint_style_encoder.get("use_extra_style_keypoints", False)
    target_training_animation_fps = data_cfg.get("target_training_animation_fps", data_cfg.animation_fps)

    # --- Training Dataset ---
    dataset = PCACachedDataSet(
        template_data_path=template_data_path,
        pca_path=pca_path,
        audio_pickle_path=str(audio_pickle_path),
        animation_dataset_path=str(animation_dataset_path),
        keypoint_pickle_path=str(landmark_pickle_path),
        metadata_path=str(metadata_path),
        source_keypoint_index_pickle=str(keypoint_index_pickle),
        use_processed_keypoint=data_cfg.use_keypoint,
        animation_fps=data_cfg.animation_fps,
        target_training_animation_fps=target_training_animation_fps,
        audio_fps=data_cfg.audio_feature_fps,
        sequence_length=data_cfg.sequence_length,
        decode_pca=data_cfg.decode_pca_in_dataset,
        disentangle_audio_exp_keypoint=data_cfg.disentangle_audio_exp_keypoint,
        disentanglement_config=data_cfg.disentanglement_config,
        device=device,
        base_audio_path=data_cfg.base_audio_path,
        target_codes=data_cfg.target_codes,
        extra_style_processed_keypoint_path=extra_style_processed_keypoint_path,
        use_extra_style_keypoints=use_extra_style_keypoints,
    )

    return dataset
