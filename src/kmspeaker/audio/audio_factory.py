# External Modules
from omegaconf import OmegaConf

# Internal Modules
from kmspeaker.audio.audio_encoder import AudioEncoder


def get_audio_model(audio_cfg):
    """
    Factory function to initialize an audio encoder based on the configuration.

    Args:
        audio_cfg (Union[DictConfig, dict]): Configuration for the audio encoder.

    Returns:
        AudioEncoder: The initialized audio encoder.
    """
    # Ensure audio_cfg is an OmegaConf object for consistent attribute access
    if isinstance(audio_cfg, dict):
        audio_cfg = OmegaConf.create(audio_cfg)

    return AudioEncoder(
        output_fps=audio_cfg.output_fps,
        seq_length=audio_cfg.data_sequence_length,
        audio_model_type=audio_cfg.audio_model_type,
        audio_model_name=audio_cfg.audio_model_name,
        combination_strategy=audio_cfg.combination_strategy,
        combination_use_softmax=audio_cfg.get("combination_use_softmax", False),
        model_dim=audio_cfg.model_dim,
        audio_model_fps=audio_cfg.audio_model_fps,
        use_cached_activations=audio_cfg.use_cached_activations,
        cached_n_hidden_layers=audio_cfg.cached_n_hidden_layers,
        cached_n_hidden_dim=audio_cfg.cached_n_hidden_dim,
        cache_dir=audio_cfg.get("cache_dir", None),
    )
