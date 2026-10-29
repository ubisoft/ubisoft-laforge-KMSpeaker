"""
Computes the Mlower / Mupper facial-region disentanglement masks (paper Supplementary Sec. B.3, eq. 4-7):

    Mlower(v) = 1                              if v in Rlower
              = 0                              if v in Rupper
              = 1 / (1 + beta * d(v, Rlower))   otherwise, d = min distance to Rlower
    Mupper = 1 - Mlower
"""
import argparse
import logging
import pickle

import numpy as np
from kmspeaker.utils.mesh_utils import load_obj_vertices

logger = logging.getLogger(__name__)


def compute_region_masks(vertices, region_indices, num_vertices, beta=1.0):
    front_face_vertex_indices = region_indices["front_face_vertices"]
    upper_front_face_vertex_indices = region_indices["eyes_forehead_region"]
    lower_front_face_vertex_indices = region_indices["mouth_jaw_region"]
    upper_back_face_vertex_indices = region_indices["upper_back_face"]
    lower_back_face_vertex_indices = region_indices["lower_back_face"]

    vertices_front_face = vertices[front_face_vertex_indices]

    dists_forehead = np.linalg.norm(
        vertices_front_face[:, None, :] - vertices[upper_front_face_vertex_indices][None, :, :], axis=2
    )
    dists_lower_face = np.linalg.norm(
        vertices_front_face[:, None, :] - vertices[lower_front_face_vertex_indices][None, :, :], axis=2
    )

    dists_forehead_min = dists_forehead.min(axis=1)
    dists_lower_face_min = dists_lower_face.min(axis=1)
    dists_lower_face_min[dists_forehead_min == 0] = 0
    dists_min_normalized = (dists_lower_face_min - dists_lower_face.min()) / (
        dists_lower_face_min.max() - dists_lower_face_min.min() + 1e-8
    )
    dists_min_normalized[dists_forehead_min == 0] = 1.0

    inverse_lower_vals = 1.0 / (1.0 + beta * dists_min_normalized)

    mask_lower = np.zeros(num_vertices, dtype=np.float32)
    mask_lower[front_face_vertex_indices] = inverse_lower_vals
    mask_lower[upper_front_face_vertex_indices] = 0.0
    mask_lower[lower_front_face_vertex_indices] = 1.0
    mask_lower[lower_back_face_vertex_indices] = 1.0
    mask_lower[upper_back_face_vertex_indices] = 0.0

    mask_upper = 1.0 - mask_lower

    return mask_lower, mask_upper


def main():
    parser = argparse.ArgumentParser(description="Compute mask_lower/mask_upper facial-region disentanglement masks.")
    parser.add_argument(
        "--region_indices_dict_path",
        default="data/dummy_region_indices_dict.pkl",
        help="Path to a pickle file containing a dict with 'front_face_vertices', 'eyes_forehead_region', "
        "'mouth_jaw_region', 'upper_back_face', and 'lower_back_face' vertex-index lists on the template mesh.",
    )
    parser.add_argument("--base_mesh_path", default="data/dataset/template/template_mesh.obj", help="Path to the template mesh (.obj).")
    parser.add_argument("--output_path", default="data/preprocessed_data/disentanglement_region_masks.pkl", help="Path to save the resulting masks (.pkl).")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse-distance decay rate.")
    parser.add_argument("--mask_key", default="mask_beta_1.0", help="Top-level key the masks are stored under, for compatibility with training logic.")
    args = parser.parse_args()

    with open(args.region_indices_dict_path, "rb") as f:
        region_indices = pickle.load(f)

    vertices = load_obj_vertices(args.base_mesh_path) # (N, 3)
    num_vertices = len(vertices)

    mask_lower, mask_upper = compute_region_masks(vertices, region_indices, num_vertices, beta=args.beta)

    out = {
        args.mask_key: {
            "audio_mask": mask_lower,
            "keypoint_mask": mask_upper,
        }
    }

    with open(args.output_path, "wb") as f:
        pickle.dump(out, f)

    logger.info(f"Masks saved to {args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
