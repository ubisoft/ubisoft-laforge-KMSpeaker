import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from kmspeaker.utils.mesh_utils import load_obj_vertices
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_keypoints_dataset(keypoints_to_save, all_keypoints, save_path_keypoints):
    """Computes mean/std across all keypoints and saves both the dataset and its statistics."""
    if all_keypoints:
        all_keypoints_concat = np.concatenate(all_keypoints, axis=0)
        mean = np.mean(all_keypoints_concat, axis=0)
        std = np.std(all_keypoints_concat, axis=0)
        keypoints_to_save["mean"] = mean
        keypoints_to_save["std"] = std

        try:
            mean_std_path = save_path_keypoints.parent / (save_path_keypoints.stem + "_statistics.pkl")
            mean_std_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mean_std_path, "wb") as f:
                pickle.dump({"mean": mean, "std": std}, f)
            logger.info(f"Mean and std saved to {mean_std_path}")
        except Exception as e:
            logger.error(f"Failed to save mean and std: {e}")

    try:
        save_path_keypoints.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_keypoints, "wb") as f:
            pickle.dump(keypoints_to_save, f)
        logger.info(f"Keypoint dataset saved to {save_path_keypoints}")
    except Exception as e:
        logger.error(f"Failed to save keypoint dataset: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-clip keypoint subsets (template-subtracted) from NPY files for multiperson "
        "data, and save the keypoints along with their mean/std statistics."
    )
    parser.add_argument("--metadata", type=str, default="data/dataset_metadata.csv", help="Path to the metadata CSV file.")
    parser.add_argument("--base_animation_path", type=str, default="data/dataset/animation", help="Path to the folder containing the per-clip animation .npy files.")
    parser.add_argument("--base_template_path", type=str, default="data/dataset/template", help="Path to the folder containing the per-identity template .obj files.")
    parser.add_argument(
        "--source_keypoints_indices_pkl",
        type=str,
        default="data/dummy_upper_and_lower_face_keypoints.pkl",
        help="Path to a pickle file containing a dict with 'upper_face_indices' and 'lower_face_indices' keypoint-index lists.",
    )
    parser.add_argument(
        "--save_path_keypoints_upper",
        type=str,
        default="data/preprocessed_data/dataset_keypoints_upper.pkl",
        help="Path to save the extracted upper-face keypoints dataset (.pkl).",
    )
    parser.add_argument(
        "--save_path_keypoints_lower",
        type=str,
        default="data/preprocessed_data/dataset_keypoints_lower.pkl",
        help="Path to save the extracted lower-face keypoints dataset (.pkl).",
    )

    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata)
    base_animation_path = Path(args.base_animation_path)
    base_template_path = Path(args.base_template_path)
    save_path_keypoints_upper = Path(args.save_path_keypoints_upper)
    save_path_keypoints_lower = Path(args.save_path_keypoints_lower)

    with open(args.source_keypoints_indices_pkl, "rb") as f:
        source_keypoints_indices = pickle.load(f)
    upper_face_indices = source_keypoints_indices["upper_face_indices"]
    lower_face_indices = source_keypoints_indices["lower_face_indices"]

    keypoints_to_save_upper = {}
    keypoints_to_save_lower = {}
    all_keypoints_upper = []
    all_keypoints_lower = []
    template_cache = {}

    logger.info("Computing keypoints...")
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Loading NPY files"):
        clip_name = row["clip_name"]
        identity = row["identity"]  # Assumes 'identity' column exists in metadata
        logger.info(f"Processing keypoints for clip: {clip_name}")

        face_npy_path = base_animation_path / f"{clip_name}.npy"
        face_data = np.load(face_npy_path)

        # Subtract identity's template, loaded from its OBJ file
        if identity not in template_cache:
            template_obj_path = base_template_path / f"{identity}.obj"
            template_cache[identity] = load_obj_vertices(template_obj_path)

        template = template_cache[identity]  # shape: (V, 3)
        if template.shape != face_data.shape[1:]:
            raise ValueError(f"Template shape {template.shape} does not match data shape {face_data.shape[1:]}")
        motion_sequence = face_data - template  # (N, V, 3) - (V, 3)

        upper_keypoints_sequence = motion_sequence[:, upper_face_indices, :]
        keypoints_to_save_upper[clip_name] = upper_keypoints_sequence
        all_keypoints_upper.append(upper_keypoints_sequence)

        lower_keypoints_sequence = motion_sequence[:, lower_face_indices, :]
        keypoints_to_save_lower[clip_name] = lower_keypoints_sequence
        all_keypoints_lower.append(lower_keypoints_sequence)

    save_keypoints_dataset(keypoints_to_save_upper, all_keypoints_upper, save_path_keypoints_upper)
    save_keypoints_dataset(keypoints_to_save_lower, all_keypoints_lower, save_path_keypoints_lower)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
