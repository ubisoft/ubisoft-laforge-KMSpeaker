import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from kmspeaker.utils.mesh_utils import load_obj_vertices
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create PCA from NPY files and save PCA model and transformed data.")
    parser.add_argument("--metadata_csv", type=str, default="data/dataset_metadata.csv", help="Path to the metadata CSV file.")
    parser.add_argument("--base_animation_path", type=str, default="data/dataset/animation", help="Path to the folder containing the per-clip animation .npy files.")
    parser.add_argument("--base_template_path", type=str, default="data/dataset/template", help="Path to the folder containing the per-identity template .obj files.")
    parser.add_argument("--save_path_pca", type=str, default="data/preprocessed_data/pca.pkl", help="Path to save the PCA model (.pkl).")
    parser.add_argument(
        "--save_path_dataset_pca_coefficients",
        type=str,
        default="data/preprocessed_data/dataset_pca_coefficients.pkl",
        help="Path to save the PCA transformed coefficients (.pkl).",
    )
    parser.add_argument("--n_components", type=int, default=128, help="Number of PCA components to keep.")

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv)
    base_animation_path = Path(args.base_animation_path)
    base_template_path = Path(args.base_template_path)
    save_path_pca = Path(args.save_path_pca)
    save_path_dataset_pca_coefficients = Path(args.save_path_dataset_pca_coefficients)

    loaded_files_info = []
    template_cache = {}

    logger.info("Loading NPY files and subtracting templates for PCA...")
    for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Loading NPY files"):
        clip_name = row["clip_name"]
        identity = row["identity"]  # Assumes 'identity' column exists in metadata

        face_npy_path = base_animation_path / f"{clip_name}.npy"
        face_sequence = np.load(face_npy_path)

        # Subtract identity's template, loaded from its OBJ file
        if identity not in template_cache:
            template_obj_path = base_template_path / f"{identity}.obj"
            template_cache[identity] = load_obj_vertices(template_obj_path)

        template = template_cache[identity]  # shape: (V, 3) or (V, D)
        if template.shape != face_sequence.shape[1:]:
            raise ValueError(f"Template shape {template.shape} does not match data shape {face_sequence.shape[1:]}")
        motion_sequence = face_sequence - template  # Broadcasting: (N, V, D) - (V, D)

        loaded_files_info.append({"filename": clip_name, "data": motion_sequence, "num_samples": motion_sequence.shape[0]})

    if not loaded_files_info:
        logger.info("No data loaded. Exiting.")
        return

    logger.info("Concatenating animations...")
    concatenated_data = np.concatenate([item["data"] for item in loaded_files_info], axis=0)
    logger.info(f"Concatenated data shape: {concatenated_data.shape}")

    logger.info("Performing PCA...")
    pca = PCA(n_components=args.n_components, whiten=False)

    num_samples = concatenated_data.shape[0]
    all_transformed_coefficients = pca.fit_transform(concatenated_data.reshape(num_samples, -1))
    logger.info(f"PCA complete. Number of components: {pca.n_components_}")
    logger.info(f"Total explained variance by {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.4f}")

    pca_model_data = {
        "mean": pca.mean_,
        "components": pca.components_,
        "singular_values": pca.singular_values_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "whiten": False,
    }
    try:
        save_path_pca.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_pca, "wb") as f:
            pickle.dump(pca_model_data, f)
        logger.info(f"PCA model saved to {save_path_pca}")
    except Exception as e:
        logger.error(f"Failed to save PCA model: {e}")

    dataset_to_save = {}
    current_idx = 0
    for item in loaded_files_info:
        num_samples_in_file = item["num_samples"]
        dataset_to_save[item["filename"]] = all_transformed_coefficients[current_idx : current_idx + num_samples_in_file]
        current_idx += num_samples_in_file

    try:
        save_path_dataset_pca_coefficients.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_dataset_pca_coefficients, "wb") as f:
            pickle.dump(dataset_to_save, f)
        logger.info(f"Dataset coefficients saved to {save_path_dataset_pca_coefficients}")
    except Exception as e:
        logger.error(f"Failed to save transformed dataset: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
