from pathlib import Path
from typing import Union

import numpy as np


def load_obj_vertices(path: Union[str, Path]) -> np.ndarray:
    """
    Loads an OBJ mesh's vertex positions, shape (V, 3), in file order.

    Parses "v x y z" lines directly rather than using a mesh loader: OBJ files can carry
    multiple material/object groups, and mesh loaders split or duplicate vertices across
    those groups, breaking the positional correspondence callers rely on (template vertices
    are indexed the same way as the per-vertex animation .npy data).
    """
    vertices = []
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            tokens = line.split()
            if tokens and tokens[0] == "v":
                vertices.append([float(x) for x in tokens[1:4]])
    return np.asarray(vertices, dtype=np.float64)
