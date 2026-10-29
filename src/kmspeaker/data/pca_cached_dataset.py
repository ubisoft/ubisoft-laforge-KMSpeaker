# Standard Library
import logging
import math
import pickle
import random
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

# External Modules
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from kmspeaker.utils.mesh_utils import load_obj_vertices

logger = logging.getLogger(__name__)


class PCACachedDataSet:
    """
    A dataset with PCA coefficients and cached audio features.

    Args:
        audio_pickle_path (str): Path to the pickled audio features.
        animation_dataset_path (str): Path to the pickled animation dataset.
        metadata_path (str): Path to the metadata CSV file.
        pca_path (Optional[str]): Path to the pickled PCA coefficients. If None, PCA is not used.
        animation_fps (int): Frames per second for the animation data.
        audio_fps (int): Frames per second for the audio data.
        sequence_length (int): Length of the sequences to be extracted (animation frames)
        decode_pca (bool): Whether to decode PCA coefficients into mesh data.
    """

    def __init__(
        self,
        audio_pickle_path: str,
        animation_dataset_path: str,
        keypoint_pickle_path: Optional[str],
        metadata_path: str,
        pca_path: Optional[str] = None,
        animation_fps: int = 30,
        audio_fps: int = 50,
        sequence_length: int = 100,
        decode_pca: bool = False,
        use_processed_keypoint: bool = False,
        disentangle_audio_exp_keypoint: bool = False,
        disentanglement_config=[0, 1, 2],
        device: str = "cpu",
        template_data_path: Optional[str] = None,
        source_keypoint_index_pickle: Optional[str] = None,
        target_training_animation_fps: Optional[int] = None,
        base_audio_path=None,
        target_codes="pca",
        extra_style_processed_keypoint_path=None,
        use_extra_style_keypoints=False,
    ): 

        logger.info("Initializing PCACachedDataSet")

        self.audio_pickle_path = audio_pickle_path
        self.animation_dataset_path = animation_dataset_path
        self.keypoint_pickle_path = keypoint_pickle_path
        self.source_keypoint_index_pickle = source_keypoint_index_pickle

        self.use_extra_style_keypoints = use_extra_style_keypoints
        self.extra_style_processed_keypoint_path = extra_style_processed_keypoint_path if use_extra_style_keypoints else None

        self.metadata_path = metadata_path
        self.base_audio_path = base_audio_path
        self.animation_fps = animation_fps
        self.target_training_animation_fps = target_training_animation_fps
        self.audio_fps = audio_fps
        self.sequence_length = sequence_length
        self.use_processed_keypoint = use_processed_keypoint
        self.template_data_path = template_data_path

        if self.use_processed_keypoint:
            logger.info("Using processed keypoint data")

        self.disentangle_audio_exp_keypoint = disentangle_audio_exp_keypoint
        self.device = device

        self.metadata = self.load_metadata()
        self.clip_names = self.metadata["clip_name"].tolist()

        self.audio_data = self.load_audio_data()
        self.animation_data = self.load_animation_data()
        if target_codes == "pca":
            self.pca_path = pca_path
            self.pca_data = self.load_pca_data()
            self.decode_pca = decode_pca
            if self.decode_pca and self.pca_data is None:
                raise ValueError("decode_pca is True but no pca_path was provided.")
            if self.decode_pca:
                self.pca_mean = self.pca_data["mean"]
                self.pca_components = self.pca_data["components"]
        else:
            self.decode_pca = False

        self.target_codes = target_codes
        self.num_keypoint = None
        self.num_extra_style_keypoint = None
        self.keypoint_indices = None
        self.extra_style_keypoint_indices = None
        self.keypoint_data, self.extra_style_keypoint_data = self.load_keypoint_data() if self.use_processed_keypoint else (None, None)
        self.disentanglement_config = disentanglement_config

        self.template = self.load_template_data()

    def resample_fps(self, data: np.ndarray, original_fps: int, target_fps: int) -> np.ndarray:
        if original_fps == target_fps or data.shape[0] <= 1:
            return data
        x = np.linspace(0, 1, num=data.shape[0])
        new_len = int(round(data.shape[0] / original_fps * target_fps))
        new_len = max(new_len, 1)
        xnew = np.linspace(0, 1, num=new_len)
        interpolator = interp1d(x, data, axis=0)
        return interpolator(xnew)

    def maybe_resample_animation_aligned_sequence(self, data: Any) -> Any:
        if self.animation_fps == self.target_training_animation_fps:
            return data
        if isinstance(data, torch.Tensor):
            return torch.tensor(self.resample_fps(data.detach().cpu().numpy(), self.animation_fps, self.target_training_animation_fps), dtype=data.dtype)
        return self.resample_fps(np.asarray(data), self.animation_fps, self.target_training_animation_fps)

    def load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_path)

    def load_audio_data(self) -> Dict[str, Any]:
        logger.info("Using precomputed audio features from pickle.")
        return torch.load(self.audio_pickle_path, map_location=torch.device("cpu"))

    def load_animation_data(self) -> Dict[str, Any]:
        with open(self.animation_dataset_path, "rb") as f:
            return pickle.load(f)

    def load_pca_data(self) -> Any:
        with open(self.pca_path, "rb") as f:
            return pickle.load(f)

    def load_template_data(self) -> Dict[str, Any]:
        template_dir = Path(self.template_data_path)
        return {identity: load_obj_vertices(template_dir / f"{identity}.obj") for identity in self.metadata["identity"].unique()}

    def load_keypoint_data(self) -> Dict[str, Any]:

        # Preprocessed upper face keypoints dataset
        with open(self.keypoint_pickle_path, "rb") as f:
            data_lmks = pickle.load(f)
        self.keypoint_mean = data_lmks["mean"]
        self.keypoint_std = data_lmks["std"]
        self.num_keypoint = self.keypoint_mean.shape[0]

        assert self.source_keypoint_index_pickle is not None, "source_keypoint_index_pickle must be set if use_processed_keypoint is True"
        with open(self.source_keypoint_index_pickle, "rb") as f:
            full_keypoint_data_idx = pickle.load(f)
        self.keypoint_indices = full_keypoint_data_idx["upper_face_indices"]
        assert self.num_keypoint == len(self.keypoint_indices), \
            f"Number of keypoints ({self.num_keypoint}) does not match length of keypoint indices ({len(self.keypoint_indices)}), ensure preprocessed keypoints and keypoint indices are consistent."

        # Preprocessed lower face keypoints dataset
        data_lmks_extra_style = None
        if self.use_extra_style_keypoints:
            with open(self.extra_style_processed_keypoint_path, "rb") as f:
                data_lmks_extra_style = pickle.load(f)
            self.extra_style_keypoint_mean = data_lmks_extra_style["mean"]
            self.extra_style_keypoint_std = data_lmks_extra_style["std"]
            self.num_extra_style_keypoint = self.extra_style_keypoint_mean.shape[0]
            self.extra_style_keypoint_indices = full_keypoint_data_idx["lower_face_indices"]
            assert self.num_extra_style_keypoint == len(self.extra_style_keypoint_indices), \
                f"Number of extra style keypoints ({self.num_extra_style_keypoint}) does not match length of extra style keypoint indices ({len(self.extra_style_keypoint_indices)}), ensure preprocessed extra style keypoints and keypoint indices are consistent."



        return data_lmks, data_lmks_extra_style

    def get_pca_decode_metadata(self) -> Dict[str, Any]:
        return {
            "pca_components": torch.tensor(self.pca_components, dtype=torch.float32, device=self.device),
            "pca_mean": torch.tensor(self.pca_mean, dtype=torch.float32, device=self.device),
        }

    def get_keypoint_metadata(self) -> Optional[Dict[str, Any]]:
        """Nested keypoint metadata (counts, indices, normalization stats) for checkpointing."""
        if not self.use_processed_keypoint:
            return None
        metadata = {
            "num_keypoint": self.num_keypoint,
            "indices": self.keypoint_indices,
            "mean": self.keypoint_mean,
            "std": self.keypoint_std,
        }
        if self.use_extra_style_keypoints:
            metadata["extra_style"] = {
                "num_keypoint": self.num_extra_style_keypoint,
                "indices": self.extra_style_keypoint_indices,
                "mean": self.extra_style_keypoint_mean,
                "std": self.extra_style_keypoint_std,
            }
        return metadata

    def __len__(self) -> int:
        return len(self.clip_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_name = self.clip_names[idx]
        clip_metadata = self.metadata[self.metadata["clip_name"] == clip_name].iloc[0]
        template = self.template[clip_metadata["identity"]]

        audio_sequence = self.audio_data[clip_metadata["audio_name"]]

        animation_sequence = self.animation_data[clip_name]
        animation_sequence = self.maybe_resample_animation_aligned_sequence(animation_sequence)

        # Determine maximum possible start frame for animation to ensure full sequence
        max_anim_start = len(animation_sequence) - (
            self.sequence_length + 2
        )  # +2 to ensure we have enough frames for the sequence after resampling - sometime the audio can be slightly shorter than animation after resampling, so this is a safeguard to avoid index out of bounds
        if max_anim_start <= 0:
            start_anim_frame = 0
        else:
            start_anim_frame = random.randint(0, int(max_anim_start))

        # Calculate corresponding audio start frame based on animation start frame
        audio_start_frame = int(start_anim_frame * (self.audio_fps / self.target_training_animation_fps))

        # Extract audio and animation sequences
        audio_end_frame = int(audio_start_frame + self.sequence_length * (self.audio_fps / self.target_training_animation_fps))
        animation_end_frame = int(start_anim_frame + self.sequence_length)

        audio_sequence = audio_sequence[audio_start_frame:audio_end_frame]
        animation_sequence = animation_sequence[start_anim_frame:animation_end_frame]

        # Convert to tensors
        audio_sequence = torch.tensor(audio_sequence, dtype=torch.float32)
        animation_sequence = torch.tensor(animation_sequence, dtype=torch.float32)

        # Use lower rounding to match int-cast slicing behavior above.
        expected_audio_len = math.floor(self.sequence_length * (self.audio_fps / self.target_training_animation_fps))  # 166 for seq len 200 at 60 fps or 83 for seq len 100 at 25 fps
        assert audio_end_frame - audio_start_frame == expected_audio_len, f"Audio window length should be {expected_audio_len} frames - now is {audio_end_frame - audio_start_frame}"

        output = {
            "audio": audio_sequence,
            "animation": animation_sequence,
            "clip_name": clip_name,
            "start_anim_frame": start_anim_frame,
            "audio_start_frame": audio_start_frame,
            "template": torch.tensor(template, dtype=torch.float32),
        }

        if self.use_processed_keypoint:
            keypoint_sequence = self.keypoint_data[clip_name]
            keypoint_sequence = self.maybe_resample_animation_aligned_sequence(keypoint_sequence)
            keypoint_sequence_window = keypoint_sequence[start_anim_frame:animation_end_frame]
            keypoint_sequence_norm_window = (keypoint_sequence_window - self.keypoint_mean) / (self.keypoint_std + 1e-8)  # Add small epsilon to avoid division by zero
            keypoint_sequence_norm_window = torch.tensor(keypoint_sequence_norm_window, dtype=torch.float32)
            output["keypoint"] = keypoint_sequence_norm_window.view(keypoint_sequence_norm_window.shape[0], -1)  # Reshape to (sequence_length, num_keypoints*3)

            ## adding extra style keypoints (lower-face keypoints added up to upper-face keypoints to be used for global style) 
            if self.use_extra_style_keypoints:
                extra_style_keypoint_sequence = self.extra_style_keypoint_data[clip_name]
                extra_style_keypoint_sequence = self.maybe_resample_animation_aligned_sequence(extra_style_keypoint_sequence)
                extra_style_keypoint_sequence_window = extra_style_keypoint_sequence[start_anim_frame:animation_end_frame]
                extra_style_keypoint_sequence_norm_window = (extra_style_keypoint_sequence_window - self.extra_style_keypoint_mean) / (self.extra_style_keypoint_std + 1e-8)  # Add small epsilon to avoid division by zero
                extra_style_keypoint_sequence_norm_window = torch.tensor(extra_style_keypoint_sequence_norm_window, dtype=torch.float32)
                output["extra_style_keypoint"] = extra_style_keypoint_sequence_norm_window.view(
                    extra_style_keypoint_sequence_norm_window.shape[0], -1
                )  # Reshape to (sequence_length, num_keypoints*3)

        # Add decoded mesh GT
        if self.target_codes == "pca" and self.decode_pca:
            # Decode PCA coefficients
            coeffs = animation_sequence.numpy()  # Convert to numpy for PCA decoding
            inverse_transformed = np.dot(coeffs, self.pca_components) + self.pca_mean
            V = self.pca_components.shape[1] // 3  # Number of vertices
            inverse_transformed_reshaped = inverse_transformed.reshape(-1, V, 3)
            inverse_transformed_reshaped += np.expand_dims(template, axis=0)
            mesh = torch.tensor(inverse_transformed_reshaped, dtype=torch.float32)
            output["mesh"] = mesh

        # Add disentanglement key: 1 = audio correct only, 2 = keypoint correct only
        if self.disentangle_audio_exp_keypoint:
            disentanglement_key = random.choice(self.disentanglement_config)
            output["disentanglement_key"] = disentanglement_key

        return output
