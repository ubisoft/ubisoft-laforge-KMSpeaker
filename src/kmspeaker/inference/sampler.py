from typing import Optional
import torch
import numpy as np
import logging
import pickle
from omegaconf import OmegaConf
import argparse
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Dict, Any, Literal
import torchaudio
from kmspeaker.utils.flow_matching_utils import ModelWrapper, ODESolver, CondOTProbPath
from kmspeaker.audio.audio_factory import get_audio_model
from kmspeaker.audio.feature_extractors import AudioFeatureExtractor
from kmspeaker.models.factory import get_model
from kmspeaker.models.style_encoder.keypoint_style_encoder import StyleEncoder
from kmspeaker.utils.mesh_utils import load_obj_vertices


logger = logging.getLogger(__name__)

class VelocityModel(ModelWrapper):
    """
    Wraps a denoiser model to convert x1 predictions into velocity fields.
    """
    def __init__(self, denoiser: torch.nn.Module, path: CondOTProbPath):
        super().__init__(model=denoiser)
        self.path = path

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        result = super().forward(x, t, **kwargs)
        x_1_prediction = result[0]
        return self.path.velocity_from_x1(x_1=x_1_prediction, x_t=x, t=t)


class CFGModelWrapper(torch.nn.Module):
    """
    Wraps a model to enable classifier-free guidance during sampling.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        cfg_scale = kwargs.get('cfg_scale', 1.0)
        keep_global_style_conditioning_no_matter_cfg_guidance_scale = kwargs.get('keep_global_style_conditioning_no_matter_cfg_guidance_scale', False)

        # Remove sampler-only kwargs before calling the model
        kwargs = {k: v for k, v in kwargs.items() if k not in [
            'cfg_scale', 'keep_global_style_conditioning_no_matter_cfg_guidance_scale', 'previous_window_style',
        ]}

        if cfg_scale == 1.0:
            # No guidance, use standard sampling
            return self.model(x, t, **kwargs)

        # Extract conditioning for conditional prediction
        conditioning = kwargs.get('conditioning', {})

        # Validate that we have the required conditioning for CFG
        if 'keypoint' not in conditioning:
            raise ValueError("CFG requires keypoint in conditioning, currently is not part of it")

        # First call: unconditional prediction (with null tokens)
        uncond_conditioning = conditioning.copy()
        uncond_conditioning['keypoint'] = torch.zeros_like(conditioning['keypoint'])
        # Use keypoint_style or keypoint_style_unconditional based on keep_global_style_conditioning_no_matter_cfg_guidance_scale
        if keep_global_style_conditioning_no_matter_cfg_guidance_scale:
            uncond_conditioning['keypoint_style'] = conditioning['keypoint_style']
        else:
            ## This will be here if ever there has been a style_encoder used. Otherwise, no need for this conditioning signal
            if 'keypoint_style_unconditional' in conditioning:
                uncond_conditioning['keypoint_style'] = conditioning['keypoint_style_unconditional']

        pred_uncond = self.model(x, t, conditioning=uncond_conditioning)

        if cfg_scale == 0.0:
            return pred_uncond

        # Second call - if needed (not 0.0): conditional prediction
        pred_cond = self.model(x, t, conditioning=conditioning)

        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

class Speech2FaceSampler:
    """
    Samples animation coefficients from pre-extracted audio features using a trained model.
    """
    def __init__(self,
                 checkpoint: dict,
                 seed: int = 42,
                 n_steps: int = 50,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 method: str = "midpoint",
                 ):
        """
        Initializes the Speech2FaceSampler.

        Args:
            checkpoint (dict): Loaded checkpoint containing model state and configuration.
            seed (int, optional): Random seed for noise initialization. Defaults to 42.
            n_steps (int, optional): Number of ODE solver steps. Defaults to 50.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
            method (str, optional): ODE integration method. Defaults to "midpoint".
        """
        self.seed = seed
        self.n_steps = n_steps
        self.device = device
        self.method = method


        self.model = None
        self.solver = None
        self.cfg = checkpoint['cfg']  # Extract configuration from checkpoint
        if isinstance(self.cfg, dict):
            self.cfg = OmegaConf.create(self.cfg)

        self._load_model(checkpoint)

    def _load_model(self, checkpoint: dict):
        """
        Initializes the model and ODE solver using the provided checkpoint.
        """
        logger.info(f"Using device: {self.device}")
        try:
            audio_encoder = get_audio_model(self.cfg.audio_encoder).to(self.device)

            keypoints_meta = checkpoint["keypoints"]
            self.num_keypoint = keypoints_meta["num_keypoint"]
            self.model = get_model(self.cfg['model'], audio_encoder, num_keypoint=self.num_keypoint)
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load keypoint style encoder if enabled
            self.keypoint_style_encoder = None
            self.num_keypoint_style = self.num_keypoint
            if self.cfg.trainer.get("use_keypoint_style_encoder", False):
                logger.info("Initializing keypoint style encoder...")
                keypoint_input_dim = self.num_keypoint * 3
                if self.cfg.model.keypoint_style_encoder.get("use_extra_style_keypoints", False):
                    num_extra_style_keypoint = keypoints_meta["extra_style"]["num_keypoint"]
                    keypoint_input_dim += num_extra_style_keypoint * 3
                    self.num_keypoint_style = self.num_keypoint + num_extra_style_keypoint
                self.keypoint_style_encoder = StyleEncoder(self.cfg.model.keypoint_style_encoder, keypoint_input_dim).to(self.device)
                if "keypoint_style_encoder_state_dict" in checkpoint:
                    self.keypoint_style_encoder.load_state_dict(checkpoint["keypoint_style_encoder_state_dict"])
                self.keypoint_style_encoder.eval()
                logger.info("Keypoint style encoder loaded.")

            # Wrap the model in VelocityModel if training target is x1_predictor
            if self.cfg.get('training_target') == 'x1_predictor':
                logger.info("Wrapping model in VelocityModel...")
                path = CondOTProbPath()
                self.model = VelocityModel(denoiser=self.model, path=path)

            # Wrap in CFG model if CFG is enabled
            self.model = CFGModelWrapper(
                model=self.model,
            )
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

        logger.info("Initializing ODE solver...")
        self.solver = ODESolver(velocity_model=self.model)

    def sample(self, audio_features: torch.Tensor, seed: int = None,
                mode: Literal["dubbing", "example_based"] = "dubbing",
                keypoint_conditioning: torch.Tensor = None,
                keypoint_conditioning_style: torch.Tensor = None,
                style_reference_keypoints: torch.Tensor = None,
                example_style_strategy: Literal["first_window", "previous_window"] = "first_window",
                ignore_global_style: bool = False, cfg_scale: float = 1.0) -> np.ndarray:
        """
        Generates animation coefficients from audio features.

        Modes:
        - "dubbing": the target's keypoints drive the frame-wise signal and the per-window
          global style (style zeroed if ignore_global_style).
        - "example_based": no frame-wise signal (cfg_scale must be 0.0). Style comes from
          style_reference_keypoints, per example_style_strategy:
            - "first_window": style of the first seq_length frames, for the whole output.
            - "previous_window" (evaluation): each window uses the seq_length frames before
              it, or the first window if there are none. The output must be longer than one
              window, and only frames from seq_length onward should be evaluated.

        Args:
            audio_features (torch.Tensor): Pre-extracted audio features, shape (batch_size, seq_len, feature_dim).
            seed (int, optional): Noise seed. Defaults to the sampler's seed.

        Returns:
            np.ndarray: The generated raw animation coefficients (pre-PCA).
        """
        assert mode in ("dubbing", "example_based"), f"Unknown mode: {mode!r}"
        if mode == "dubbing":
            assert cfg_scale > 0.0, "mode='dubbing' requires cfg_scale > 0.0"
        else:
            assert cfg_scale == 0.0, "mode='example_based' requires cfg_scale == 0.0"
            assert style_reference_keypoints is not None, "mode='example_based' requires style_reference_keypoints"
            assert example_style_strategy in ("first_window", "previous_window"), \
                f"Unknown example_style_strategy: {example_style_strategy!r}"

        # example_based only runs the unconditional CFG branch, so style must be kept there
        keep_global_style_conditioning_no_matter_cfg_guidance_scale = (mode == "example_based")

        current_seed = seed if seed is not None else self.seed
        logger.info(f"Generating initial noise with seed: {current_seed}")
        torch.manual_seed(current_seed)

        # Ensure audio_features is on the correct device
        audio_features = audio_features.to(self.device)

        batch_size = audio_features.shape[0]
        total_audio_len = audio_features.shape[1]

        # Calculate corresponding animation length
        audio_fps = self.cfg['audio_encoder']['audio_model_fps']
        anim_fps = self.cfg['data']['animation_fps']
        total_anim_len = int(np.round(total_audio_len * anim_fps / audio_fps))

        # Expected single-chunk lengths
        expected_anim_len = self.cfg['data']['sequence_length']

        input_dim = self.cfg['model']["pca_dim"]

        # Use full length for initialization
        x_init = torch.randn((batch_size, total_anim_len, input_dim), device=self.device)

        logger.info(f"Solving ODE for {self.n_steps} steps with sequence length {total_anim_len}...")
        T = torch.linspace(0, 1, self.n_steps).to(self.device)

        # Validate CFG compatibility
        if cfg_scale != 1.0:
            cfg_probability = self.cfg['trainer'].get('cfg_probability', 0.0)
            if cfg_probability <= 0:
                logger.warning(f"CFG scale {cfg_scale} requested, but model was not trained with CFG")

        if keypoint_conditioning is None:
            assert cfg_scale == 0.0, "When no keypoint conditioning for frame wise signal is provided, CFG scale must be 0.0, right?"
            keypoint_conditioning = torch.zeros((batch_size, total_anim_len, self.num_keypoint, 3), device=self.device)
            keypoint_conditioning_style = torch.zeros((batch_size, total_anim_len, self.num_keypoint_style, 3), device=self.device)

        try:
            with torch.no_grad():
                conditioning = {'audio': audio_features}

                if keypoint_conditioning is not None:
                    keypoint_conditioning = keypoint_conditioning.to(self.device).reshape(keypoint_conditioning.shape[0], keypoint_conditioning.shape[1], -1) # Reshape to (sequence_length, num_keypoints*3)
                    keypoint_conditioning_style = keypoint_conditioning_style.to(self.device).reshape(keypoint_conditioning_style.shape[0], keypoint_conditioning_style.shape[1], -1) # Reshape to (sequence_length, num_keypoints*3)
                    conditioning["keypoint"] = keypoint_conditioning[:, :total_anim_len] # Takes at least as long keypoint condition as possible regarding audio
                    conditioning["keypoint_style_to_proc"] = keypoint_conditioning_style[:, :total_anim_len]

                    if conditioning["keypoint"].shape[1] - total_anim_len < 2:
                        total_anim_len = conditioning["keypoint"].shape[1]
                        x_init = x_init[:, :total_anim_len]

                    assert conditioning["keypoint"].shape[1] == total_anim_len, f"keypoint conditioning sequence lenght is different from total_anim_len  ({conditioning['keypoint'].shape[1]} vs {total_anim_len})"
                    assert conditioning["keypoint_style_to_proc"].shape[1] == total_anim_len, f"keypoint_style conditioning sequence lenght is different from total_anim_len  ({conditioning['keypoint_style_to_proc'].shape[1]} vs {total_anim_len})"

                if self.keypoint_style_encoder:
                    keypoint_style_zero = self.keypoint_style_encoder(torch.zeros_like(keypoint_conditioning_style[:, :expected_anim_len]), do_sample=True)
                    conditioning["keypoint_style_unconditional"] = keypoint_style_zero

                previous_window_style = False
                if mode == "dubbing":
                    # Style comes from the target's own keypoints, per window. ignore_global_style
                    # zeroes style only, not the frame-wise "keypoint" signal.
                    if ignore_global_style and self.keypoint_style_encoder:
                        conditioning["keypoint_style"] = conditioning["keypoint_style_unconditional"]
                        # Full length, since the chunked path slices it per chunk
                        conditioning["keypoint_style_to_proc"] = torch.zeros_like(keypoint_conditioning_style[:, :total_anim_len])
                else:  # mode == "example_based"
                    assert self.keypoint_style_encoder is not None, "mode='example_based' requires a keypoint style encoder"
                    previous_window_style = (example_style_strategy == "previous_window")
                    if previous_window_style:
                        # The first window is styled by itself, so evaluation needs frames beyond it
                        assert total_anim_len > expected_anim_len, (
                            "example_style_strategy='previous_window' requires the target animation to be "
                            f"longer than a single window ({expected_anim_len} frames), got {total_anim_len} frames."
                        )
                        assert style_reference_keypoints.shape[0] >= total_anim_len, (
                            "example_style_strategy='previous_window' requires style_reference_keypoints to be "
                            f"at least as long as the output ({style_reference_keypoints.shape[0]} < {total_anim_len})."
                        )
                        # Style is encoded per chunk in chunked_inference.py; "keypoint_style" is
                        # left unset on purpose so that path runs.
                        ref_len = style_reference_keypoints.shape[0]
                        conditioning["keypoint_style_to_proc"] = style_reference_keypoints.reshape(
                            ref_len, -1).unsqueeze(0)
                    else:
                        # "first_window": encode the first window once, for the whole generation
                        style_window = style_reference_keypoints[:expected_anim_len]
                        conditioning["keypoint_style"] = self.keypoint_style_encoder(
                            style_window.reshape(style_window.shape[0], -1).unsqueeze(0), do_sample=True)

                ## Prepare kwargs:
                sample_kwargs = {
                    "integration_schedule": T,
                    "initial_noise": x_init,
                    "conditioning": conditioning,
                    "solver_method": self.method,
                    "fixed_step_size": 1.0 / self.n_steps,
                    "cfg_scale": cfg_scale,
                    "keep_global_style_conditioning_no_matter_cfg_guidance_scale": keep_global_style_conditioning_no_matter_cfg_guidance_scale,
                    "previous_window_style": previous_window_style,
                }

                # Create a chunked model wrapper if sequence is longer than expected
                if total_anim_len > expected_anim_len:
                    from kmspeaker.utils.chunked_inference import ChunkedModelWrapper
                    chunked_model = ChunkedModelWrapper(self.model, self.cfg, self.keypoint_style_encoder)
                    # Temporarily replace the solver's velocity model
                    original_model = self.solver.velocity_model
                    self.solver.velocity_model = chunked_model

                    sol = self.solver.sample(**sample_kwargs)

                    # Restore original model
                    self.solver.velocity_model = original_model
                else:
                    # Single window: dubbing encodes style from its own (possibly zeroed)
                    # keypoints; example_based already set its style above.
                    if mode == "dubbing" and self.keypoint_style_encoder and keypoint_conditioning_style is not None:
                        logger.info("Processing keypoint style using keypoint style encoder...")
                        keypoint_style = self.keypoint_style_encoder(conditioning["keypoint_style_to_proc"], do_sample=True)
                        conditioning["keypoint_style"] = keypoint_style
                    sample_kwargs["conditioning"] = conditioning
                    sol = self.solver.sample(**sample_kwargs)

                raw_coeffs = sol.cpu().numpy()

        except Exception as e:
            logger.error(f"Error solving ODE: {e}")
            raise

        logger.info("Raw coefficient sampling complete.")
        return raw_coeffs

# --- Orchestrator Class ---
class SpeechToFacePipeline:
    """
    Orchestrates the full pipeline from WAV to PCA-transformed animation coefficients.
    """
    def __init__(self,
                 checkpoint_path: str,
                 seed: int = 42,
                 n_steps: int = 50,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 method: str = "midpoint",
                 ):


        self.device = device
        self.seed = seed
        self.n_steps = n_steps
        self.method = method

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.cfg = checkpoint['cfg']
        if isinstance(self.cfg, dict):
            self.cfg = OmegaConf.create(self.cfg)


        self.data_root = Path(self.cfg.data_root)
        keypoints_meta = checkpoint.get("keypoints") or {}
        self.keypoint_indices = keypoints_meta.get("indices")
        self.keypoint_mean = keypoints_meta.get("mean")
        self.keypoint_std = keypoints_meta.get("std")
        extra_style_meta = keypoints_meta.get("extra_style") or {}
        self.extra_style_keypoint_indices = extra_style_meta.get("indices")
        self.extra_style_keypoint_mean = extra_style_meta.get("mean")
        self.extra_style_keypoint_std = extra_style_meta.get("std")

        self.target_codes= self.cfg.trainer.get("target_codes", "pca") # Default is PCA when not defined

        self.model_type = self.cfg.model.type
        self.w2vbert_model_name = self.cfg.audio_encoder.audio_model_name


        logger.info(f"Initializing Audio feature with model: {self.w2vbert_model_name}")

        self.feature_extractor = AudioFeatureExtractor(model_name=self.w2vbert_model_name, cache_dir=self.cfg.audio_encoder.get("cache_dir", None)).to(self.device)
        self.feature_extractor.eval()

        self.use_processed_keypoint = self.cfg.data.use_keypoint

        self.sampler = Speech2FaceSampler(
            checkpoint=checkpoint,
            seed=seed,
            n_steps=n_steps,
            device=device,
            method=self.method,
        )

        self.pca_data = None
        if self.target_codes == "pca":
            try:
                self.pca_path = self.data_root / self.cfg.data.pca_path
            except Exception as e:
                self.pca_path = self.data_root / self.cfg.data.animation_data
            self.pca_path = str(self.pca_path)
            self.pca_data = self._load_pca()

        self.target_fps = self.cfg.data.animation_fps

        self.keypoint_data = None

    def _load_pca(self):
        """
        Loads PCA data from the specified path and returns it as a dictionary.

        Returns:
            dict: A dictionary containing PCA-related data, including mean, components,
                    whiten, and explained_variance.
        """
        logger.info(f"Loading PCA from: {self.pca_path}")
        try:
            with open(self.pca_path, 'rb') as f:
                pca_model_data = pickle.load(f)

            # Ensure all data is in numpy format
            if isinstance(pca_model_data['mean'], torch.Tensor):
                pca_model_data['mean'] = pca_model_data['mean'].cpu().numpy()
            if isinstance(pca_model_data['components'], torch.Tensor):
                pca_model_data['components'] = pca_model_data['components'].cpu().numpy()

            return pca_model_data
        except Exception as e:
            logger.error(f"Error loading PCA: {e}")
            raise

    def load_style_reference_sequence(self, style_reference_sequence_path, template_target: np.ndarray, target_animation_fps: Optional[int] = 60):
        """
        Loads the example-based style reference clip and returns its style keypoints.
        """
        if style_reference_sequence_path is not None:
            logger.info(f"Loading style reference sequence: {style_reference_sequence_path}")
            style_reference_sequence_path = Path(style_reference_sequence_path)
            style_reference_sequence = np.load(style_reference_sequence_path)

            if target_animation_fps is not None and target_animation_fps != 60:
                logger.info(f"Resampling style reference sequence from {target_animation_fps} FPS to 60 FPS")
                style_reference_sequence = self.resample_fps(style_reference_sequence, original_fps=target_animation_fps)

            _, style_reference_keypoints = self.prepare_processed_keypoint(
                target_exp_sequence=style_reference_sequence,
                neutral_template=template_target,
                extra_style_keypoints=self.cfg.model.keypoint_style_encoder.get("use_extra_style_keypoints", False)
                )
            return style_reference_keypoints
        return None

    def load_dubbing_target_sequence(self, dubbing_target_sequence_path: str, scale: float, shift: tuple, target_animation_fps: Optional[int] = 60) -> np.ndarray:
        """
        Loads the frame-aligned dubbing target performance.
        """
        dubbing_target_sequence = None
        if dubbing_target_sequence_path is not None:
            logger.info(f"Loading dubbing target sequence: {dubbing_target_sequence_path}")
            dubbing_target_sequence_path = Path(dubbing_target_sequence_path)
            dubbing_target_sequence = np.load(dubbing_target_sequence_path)
            logger.info(f"Applying scale {scale} and shift {shift} to dubbing target sequence")
            dubbing_target_sequence = (dubbing_target_sequence * scale) + np.array(shift)

            if target_animation_fps is not None and target_animation_fps != 60:
                logger.info(f"Resampling dubbing target sequence from {target_animation_fps} FPS to 60 FPS")
                dubbing_target_sequence = self.resample_fps(dubbing_target_sequence, original_fps=target_animation_fps)
        return dubbing_target_sequence

    def prepare_processed_keypoint(self, target_exp_sequence, neutral_template=None, extra_style_keypoints: bool=False) -> Optional[torch.Tensor]:
        """
        Prepares landmark data for conditioning if a target expression sequence is provided.
        """
        if target_exp_sequence is not None:
            assert self.keypoint_indices is not None, "checkpoint['keypoints']['indices'] must be set if target_exp_sequence is specified"
            source_landmarks = self.keypoint_indices

            logger.info("Getting deltas prior to landmark extraction and normalization")
            target_exp_sequence = target_exp_sequence - neutral_template

            logger.info("Preparing target landmark sequence - normalization using stats stored in the checkpoint")
            landmarks = target_exp_sequence[:, source_landmarks, :]  # Select only the specified landmarks (seq_len, num_landmarks, 3)
            landmarks = (landmarks - self.keypoint_mean) / self.keypoint_std

            landmarks_style = landmarks.copy()
            if extra_style_keypoints:
                assert self.extra_style_keypoint_indices is not None, "checkpoint['keypoints']['extra_style']['indices'] must be set if extra_style_keypoints is requested"
                extra_style_keypoint_indices = self.extra_style_keypoint_indices
                logger.info(f"Using extra style keypoint indices: {extra_style_keypoint_indices}")

                # Extract extra style keypoints
                extra_style_keypoints = target_exp_sequence[:, extra_style_keypoint_indices, :]  # (seq_len, num_extra_style_keypoints, 3)
                extra_style_keypoints = (extra_style_keypoints - self.extra_style_keypoint_mean) / self.extra_style_keypoint_std

                landmarks_style = np.concatenate([landmarks, extra_style_keypoints], axis=1)

            return torch.from_numpy(landmarks).to(torch.float32).to(self.device), \
                   torch.from_numpy(landmarks_style).to(torch.float32).to(self.device) if extra_style_keypoints is not None else torch.from_numpy(landmarks).to(torch.float32).to(self.device)
        return None

    def resample_fps(self, data, original_fps):
        """
        Resample data to a target FPS using linear interpolation.

        Args:
            data (np.ndarray): The data to be resampled.
            original_fps (int): The original FPS of the data.

        Returns:
            np.ndarray: The resampled data.
        """

        x = np.linspace(0, 1, num=data.shape[0])
        xnew = np.linspace(0, 1, num=int(round(data.shape[0] / original_fps * self.target_fps)))
        interpolator = interp1d(x, data, axis=0)
        return interpolator(xnew)

    def process_wav(self, template_target, template_decoding, wav_path: str, seed: int = None,
                    mode: Literal["dubbing", "example_based"] = "dubbing",
                    cfg_guidance_scale: Optional[float] = None,
                    dubbing_target_sequence_path: Optional[str] = None,
                    style_reference_sequence_path: Optional[str] = None,
                    example_style_strategy: Literal["first_window", "previous_window"] = "first_window",
                    scale:Optional[float]=1, shift:Optional[tuple]=(0.,0.,0.),
                    ignore_global_style: bool = False, target_animation_fps: Optional[int] = 60) -> dict:
        """
        Processes a single WAV file to generate animation coefficients.

        Args:
            wav_path (str): Path to the input WAV file.
            seed (int, optional): Random seed for noise initialization for the sampler.
                                    Defaults to None (uses the pipeline's default seed).
            mode: "dubbing" (needs dubbing_target_sequence_path) or "example_based"
                (needs style_reference_sequence_path).
            cfg_guidance_scale: Defaults to 1.0 for "dubbing", must be 0.0 for "example_based".
            target_animation_fps: FPS of the target/style sequence, resampled to 60 if different.

        Returns:
            dict: "mesh" (decoded vertices, or raw coefficients without PCA), "raw_coefficients" and "fps".
        """
        if mode == "dubbing":
            assert dubbing_target_sequence_path is not None, "mode='dubbing' requires dubbing_target_sequence_path"
            assert style_reference_sequence_path is None, (
                "mode='dubbing' does not take style_reference_sequence_path; style comes from dubbing_target_sequence_path."
            )
            if cfg_guidance_scale is None:
                cfg_guidance_scale = 1.0
            assert cfg_guidance_scale > 0.0, "mode='dubbing' requires cfg_guidance_scale > 0.0"
        elif mode == "example_based":
            assert style_reference_sequence_path is not None, "mode='example_based' requires style_reference_sequence_path"
            assert dubbing_target_sequence_path is None, "mode='example_based' does not take dubbing_target_sequence_path"
            if cfg_guidance_scale is None:
                cfg_guidance_scale = 0.0
            assert cfg_guidance_scale == 0.0, "mode='example_based' requires cfg_guidance_scale == 0.0"
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # 1. Load audio
        logger.info(f"Loading audio from: {wav_path}")
        try:
            import librosa
            waveform, sr = librosa.load(wav_path, sr=None)
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
            if waveform_tensor.ndim > 1 and waveform_tensor.shape[1] > 1:
                waveform_tensor = waveform_tensor[:, 0]  # Take the first channel

            # Resample to 16kHz if needed
            target_sr = 16000
            if sr != target_sr:
                waveform_tensor = waveform_tensor.unsqueeze(0) if waveform_tensor.ndim == 1 else waveform_tensor.T.unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform_tensor = resampler(waveform_tensor)
                waveform_tensor = waveform_tensor.squeeze(0)
                sr = target_sr

            waveform_tensor = waveform_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

        # 2. Extract features
        logger.info("Extracting audio features...")
        try:
            with torch.no_grad():
                audio_features = self.feature_extractor(waveform_tensor, sampling_rate=sr)
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            raise

        # Ensure audio_features are not longer than the model's expected seq_len
        seq_length = self.cfg.data.sequence_length
        audio_fps = self.cfg.audio_encoder.audio_model_fps
        anim_fps = self.cfg.data.animation_fps

        expected_audio_len = int(np.round(seq_length * audio_fps / anim_fps))
        seq_length_audio = seq_length
        actual_audio_length = audio_features.shape[1]
        actual_audio_length_animation = int(np.round(actual_audio_length * anim_fps / audio_fps))
        if audio_features.shape[1] < expected_audio_len:
            seq_length_audio = int(np.round(audio_features.shape[1] * anim_fps / audio_fps))

            logger.warning(f"Padding audio features from {audio_features.shape[1]} to minimum length {expected_audio_len}")
            pad_len = expected_audio_len - audio_features.shape[1]
            pad = torch.zeros((audio_features.shape[0], pad_len, audio_features.shape[2], audio_features.shape[3]), device=audio_features.device, dtype=audio_features.dtype)
            audio_features = torch.cat([audio_features, pad], dim=1)
        else:
            logger.info(f"Processing audio with {audio_features.shape[1]} frames (expected minimum: {expected_audio_len})")


        ###### PREPARE KEYPOINT CONDITIONING IF NEEDED ######
        dubbing_target_sequence = self.load_dubbing_target_sequence(dubbing_target_sequence_path, scale, shift, target_animation_fps=target_animation_fps) if dubbing_target_sequence_path is not None else None
        style_reference_keypoints = self.load_style_reference_sequence(style_reference_sequence_path, template_target, target_animation_fps=target_animation_fps) if style_reference_sequence_path is not None else None

        if self.use_processed_keypoint and dubbing_target_sequence is not None:
            self.keypoint_data, self.keypoint_style_data = self.prepare_processed_keypoint(
                target_exp_sequence=dubbing_target_sequence,  # Ensure we only take the face vertices for keypoint conditioning
                neutral_template=template_target,
                extra_style_keypoints=self.cfg.model.keypoint_style_encoder.get("use_extra_style_keypoints", False)
            )
        else:
            self.keypoint_data, self.keypoint_style_data = None, None

        # Ensure keypoint_data is not longer than the model's expected seq_len - Pad if necessary - Crop if necessary
        if self.keypoint_data is not None:
            self.keypoint_data = self.keypoint_data.unsqueeze(0)
            if self.keypoint_data.shape[1] < seq_length:
                pad_len = seq_length - self.keypoint_data.shape[1]
                pad = torch.zeros((self.keypoint_data.shape[0], pad_len, self.keypoint_data.shape[2], self.keypoint_data.shape[3]), device=self.keypoint_data.device, dtype=self.keypoint_data.dtype)
                self.keypoint_data = torch.cat([self.keypoint_data, pad], dim=1)

        if self.keypoint_style_data is not None:
            self.keypoint_style_data = self.keypoint_style_data.unsqueeze(0)
            if self.keypoint_style_data.shape[1] < seq_length:
                pad_len = seq_length - self.keypoint_style_data.shape[1]
                pad = torch.zeros((self.keypoint_style_data.shape[0], pad_len, self.keypoint_style_data.shape[2], self.keypoint_style_data.shape[3]), device=self.keypoint_style_data.device, dtype=self.keypoint_style_data.dtype)
                self.keypoint_style_data = torch.cat([self.keypoint_style_data, pad], dim=1)

        if mode == "dubbing":
            ## At all time reduce to the length of the audio (in frames 60fps)
            dubbing_target_sequence = dubbing_target_sequence[:actual_audio_length_animation]

        # 3. Sample raw coefficients
        logger.info("Sampling...")
        with torch.no_grad():
            raw_coeffs = self.sampler.sample(audio_features=audio_features,
                                             seed=seed,
                                             mode=mode,
                                             keypoint_conditioning=self.keypoint_data,
                                             keypoint_conditioning_style=self.keypoint_style_data,
                                             style_reference_keypoints=style_reference_keypoints,
                                             example_style_strategy=example_style_strategy,
                                             ignore_global_style=ignore_global_style,
                                             cfg_scale=cfg_guidance_scale,
                                            )

        if seq_length_audio != seq_length:
            logger.info(f"Trimming raw coefficients from {raw_coeffs.shape[1]} to {seq_length_audio} frames.")
            raw_coeffs = raw_coeffs[:, :seq_length_audio, :]

        def _make_result(mesh_vertices):
            return {
                'mesh':             mesh_vertices,
                'raw_coefficients': raw_coeffs.squeeze(),
                'fps':              self.target_fps,
            }

        # --- Decoding block for PCA ---
        if self.target_codes == "pca" and self.pca_data:
            logger.info("Applying inverse PCA transformation...")
            if self.pca_data.get('whiten', False):
                raw_coeffs *= np.sqrt(self.pca_data['explained_variance'])
            mesh = np.dot(raw_coeffs, self.pca_data['components']) + self.pca_data['mean']
            mesh = mesh.reshape(raw_coeffs.shape[0], raw_coeffs.shape[1], -1, 3)
            if template_decoding is not None:
                mesh = mesh + template_decoding[None, ...]
            mesh_vertices = mesh.squeeze()
            logger.info("PCA transformation complete.")
            return _make_result(mesh_vertices)
        else:
            return _make_result(raw_coeffs.squeeze())

class InferencePipeline:
    """
    Class-based interface for speech-to-face inference.
    Initializes the pipeline once and allows multiple inference calls via process_clip().
    """

    def __init__(
        self,
        checkpoint_path: str,
        seed: int = 42,
        n_steps: int = 50,
        method: str = "midpoint",
    ):
        """
        Initialize the inference pipeline.

        Args:
            checkpoint_path: Path to the model checkpoint.
            seed: Random seed for noise initialization.
            n_steps: Number of ODE solver steps.
            method: ODE integration method (e.g., "midpoint", "euler", "rk4").
        """
        self.pipeline = SpeechToFacePipeline(
            checkpoint_path=checkpoint_path,
            seed=seed,
            n_steps=n_steps,
            method=method,
        )

    def process_clip(
        self,
        wav_path: str,
        output_folder: str,
        target_neutral: str,
        mode: Literal["dubbing", "example_based"] = "dubbing",
        decoding_neutral: Optional[str] = None,
        output_name: Optional[str] = None,
        dubbing_target_sequence: Optional[str] = None,
        style_reference_sequence: Optional[str] = None,
        example_style_strategy: Literal["first_window", "previous_window"] = "first_window",
        cfg_guidance_scale: Optional[float] = None,
        ignore_global_style: bool = False,
        scale: float = 1.0,
        shift: tuple = (0., 0., 0.),
        target_animation_fps: Optional[int] = 60,
    ) -> dict:
        """
        Process a single audio clip and generate facial animation.

        Args:
            wav_path: Path to the input wav file.
            output_folder: Directory to save outputs.
            target_neutral: Path to target neutral .obj file for template.
            mode: "dubbing" or "example_based", see Speech2FaceSampler.sample.
            decoding_neutral: Path to decoding neutral .obj file. If omitted, uses target_neutral.
            output_name: Optional name for output file.
            dubbing_target_sequence: Dubbing-only. Frame-aligned target performance path.
            style_reference_sequence: Example-based-only. Style reference clip path.
            example_style_strategy: Example-based-only. See Speech2FaceSampler.sample.
            cfg_guidance_scale: Defaults to 1.0 for "dubbing", must be 0.0 for "example_based".
            ignore_global_style: Dubbing-only. Zeroes the style signal only.
            scale: Scale factor for target sequence.
            shift: Shift tuple (x, y, z) for target sequence.
            target_animation_fps: FPS of the target/style sequence, resampled to 60 if different.

        Returns:
            dict: Dictionary containing output paths and results.
        """

        # Load templates
        template_target = load_obj_vertices(target_neutral) if target_neutral else None
        template_decoding = load_obj_vertices(decoding_neutral) if decoding_neutral else template_target

        result = self.pipeline.process_wav(
            template_target=template_target,
            template_decoding=template_decoding,
            wav_path=wav_path,
            mode=mode,
            cfg_guidance_scale=cfg_guidance_scale,
            dubbing_target_sequence_path=dubbing_target_sequence,
            style_reference_sequence_path=style_reference_sequence,
            example_style_strategy=example_style_strategy,
            scale=scale,
            shift=shift,
            ignore_global_style=ignore_global_style,
            target_animation_fps=target_animation_fps
        )

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if output_name is not None:
            output_npy = output_dir / Path(output_name).with_suffix('.npy')
        else:
            target_lmks_clip_name = Path(dubbing_target_sequence).stem if dubbing_target_sequence is not None else None
            audio_clip_name = Path(wav_path).stem

            if target_lmks_clip_name is not None:
                long_name = f"{audio_clip_name}_guided_by_{target_lmks_clip_name}.npy"
            else:
                long_name = f"{audio_clip_name}.npy"

            output_npy = output_dir / long_name

        # Save the animated mesh (face only - eyes and teeth are never part of the output)
        logger.info(f"Saving face mesh vertices to: {output_npy}")
        np.save(output_npy, result['mesh'])

        return {
            'mesh_path': str(output_npy),
            'result': result,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Sample from a trained speech2face model using the pipeline.")

    # Input / output files
    parser.add_argument("--wav_path", type=str, required=True, help="Path to the input wav file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the output numpy array.")
    parser.add_argument("--output_name", type=str, required=False, help="Path to save the output numpy array.")
    parser.add_argument("--dubbing_target_sequence", type=str, default=None, help="Dubbing-only. Frame-aligned target performance path used for lip-sync conditioning and (unless --ignore_global_style) global style.")
    parser.add_argument("--target_animation_fps", type=int, default=None, help="The FPS of the target animation.")
    parser.add_argument("--target_neutral", type=str, default=None,
                        help="Full path to a target neutral .obj file used as template.")
    parser.add_argument("--decoding_neutral", type=str, default=None,
                        help="Optional path to neutral template used for decoding. If omitted, uses --target_neutral.")

    # Model related parameters
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise initialization (default: 42).")
    parser.add_argument("--n_steps", type=int, default=50, help="Number of ODE solver steps (default: 50).")
    parser.add_argument("--method", type=str, default="midpoint", help="ODE integration method (default: midpoint).")

    # Control parameters
    parser.add_argument("--mode", type=str, required=True, choices=["dubbing", "example_based"], help="Prediction mode: 'dubbing' (requires --dubbing_target_sequence) or 'example_based' (requires --style_reference_sequence).")
    parser.add_argument("--cfg_guidance_scale", default=None, type=float, help="Guidance scale for the sampler. Defaults to 1.0 for --mode=dubbing, forced to 0.0 for --mode=example_based.")
    parser.add_argument("--ignore_global_style", action="store_true", help="Dubbing-only. If set, will pass a zero signal to the style encoder.")
    parser.add_argument("--style_reference_sequence", type=str, default=None, help="Example-based-only. Path to a style reference clip to use as global style for the full generation.")
    parser.add_argument("--example_style_strategy", type=str, default="first_window", choices=["first_window", "previous_window"], help="Example-based-only. How to derive the global style from --style_reference_sequence.")

    # Optional parameters
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--shift', type=str, default="(0, 0, 0)")

    args = parser.parse_args()

    args.shift = tuple(map(float, args.shift.strip('()').split(',')))

    # Create the inference pipeline (model loaded once, template-independent)
    inference = InferencePipeline(
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        n_steps=args.n_steps,
        method=args.method,
    )

    # Process clip (template provided per-call)
    output = inference.process_clip(
        wav_path=args.wav_path,
        output_folder=args.output_folder,
        target_neutral=args.target_neutral,
        mode=args.mode,
        decoding_neutral=args.decoding_neutral,
        output_name=args.output_name,
        dubbing_target_sequence=args.dubbing_target_sequence,
        style_reference_sequence=args.style_reference_sequence,
        example_style_strategy=args.example_style_strategy,
        cfg_guidance_scale=args.cfg_guidance_scale,
        ignore_global_style=args.ignore_global_style,
        scale=args.scale,
        shift=args.shift,
        target_animation_fps=args.target_animation_fps,
    )

    logger.info(f"Output saved to: {output.get('mesh_path')}")
    logger.info("Done.")
