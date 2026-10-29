# External Modules
import torch.nn as nn

# Internal Modules
from kmspeaker.audio.feature_extractors import AudioFeatureExtractor

# Local Modules
from .layer_combiner import ScalarLayerCombiner
from .temporal_resampler import LinearResampler


class AudioEncoder(nn.Module):
    """
    Process audio, combine hidden states from multiple layers, and resample to target FPS.

    Args:
        output_fps (int): Target FPS for the final audio embeddings (e.g. animation FPS).
        audio_model_type (str): Type of audio model to use (e.g. "w2v-bert").
        audio_model_name (str): Name of the pre-trained audio model to load.
        combination_strategy (str): Strategy for combining hidden layers (e.g. "scalar").
        model_dim (int): Dimension of the final audio embeddings.
        audio_model_fps (int): FPS of the audio model's output (e.g. 50 for w2v-bert).
        use_cached_activations (bool): If True, uses pre-computed hidden states instead of raw audio.
        cached_n_hidden_layers (int, optional): Number of hidden layers in the cached activations.
        cached_n_hidden_dim (int, optional): Dimension of the hidden states in the cached activations.
        cache_dir (str, optional): Download cache directory for the audio model.

    """

    def __init__(
        self,
        output_fps,
        seq_length,  # Animation sequence length
        audio_model_type="w2v-bert",
        audio_model_name="facebook/w2v-bert-2.0",
        combination_strategy="scalar",
        combination_use_softmax=False,
        model_dim=512,
        audio_model_fps=50,
        use_cached_activations=False,
        cached_n_hidden_layers=None,
        cached_n_hidden_dim=None,
        cache_dir=None,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.use_cached_activations = use_cached_activations
        self.feature_extractor = None
        self.num_hidden_layers_from_extractor = None
        self.n_hidden_dim_from_extractor = None

        if not self.use_cached_activations:
            if audio_model_type == "w2v-bert":
                self.feature_extractor = AudioFeatureExtractor(model_name=audio_model_name, cache_dir=cache_dir)
                self.num_hidden_layers_from_extractor = self.feature_extractor.num_hidden_layers
                self.n_hidden_dim_from_extractor = self.feature_extractor.n_hidden_dim
            else:
                raise ValueError(f"Unsupported audio_model_type: {audio_model_type}")
        else:
            self.num_hidden_layers_from_extractor = cached_n_hidden_layers
            self.n_hidden_dim_from_extractor = cached_n_hidden_dim

        if combination_strategy == "scalar":
            if self.num_hidden_layers_from_extractor is None or self.n_hidden_dim_from_extractor is None:
                raise ValueError(
                    "Cannot initialize ScalarLayerCombiner without num_hidden_layers and n_hidden_dim. "
                    "Ensure feature_extractor is initialized or provide these values when using cached activations."
                )
            self.combiner = ScalarLayerCombiner(
                num_layers=self.num_hidden_layers_from_extractor, input_dim=self.n_hidden_dim_from_extractor, output_dim=model_dim, use_softmax=combination_use_softmax
            )
        else:
            raise ValueError(f"Unsupported combination_strategy: {combination_strategy}")

        self.resampler = LinearResampler(input_fps=audio_model_fps, output_fps=output_fps)

    def forward(self, audio_or_hidden_states, sampling_rate=None):
        """
        Processes raw audio or pre-computed hidden states.

        Args:
            audio_or_hidden_states (torch.Tensor or list):
                If use_cached_activations is False: Batch of raw audio signals.
                If use_cached_activations is True: Pre-computed stacked hidden states
                                                   (N x T_audio_model x n_layers x n_dim).
            sampling_rate (int, optional): Sampling rate of the audio.
                                           Required if use_cached_activations is False.

        Returns:
            torch.Tensor: Final audio embeddings at the target animation FPS
                          (N x T_animation_fps x model_dim).
        """
        if not self.use_cached_activations:
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not initialized. This should not happen if use_cached_activations is False.")
            if sampling_rate is None:
                raise ValueError("sampling_rate must be provided when processing raw audio.")
            # Input: Raw audio (N x T_orig) or list of raw audio
            # Output: Stacked hidden states (N x T_audio_fps x n_hidden_layers x n_hidden_dim)
            hidden_states = self.feature_extractor(audio_or_hidden_states, sampling_rate)
        else:
            # Input: Pre-computed raw hidden states (N x T_audio_fps x n_hidden_layers x n_hidden_dim)
            hidden_states = audio_or_hidden_states

        # Input: Stacked hidden states (N x T_audio_fps x n_hidden_layers x n_hidden_dim)
        # Output: Combined features (N x T_audio_fps x model_dim)
        combined_features = self.combiner(hidden_states)

        # Input: Combined features (N x T_current_fps x model_dim)
        # Output: Resampled features (N x T_animation_fps x model_dim)
        # Note: T_current_fps here is T_audio_fps, which corresponds to audio_model_fps
        resampled_features = self.resampler(combined_features, target_length=self.seq_length)

        return resampled_features
