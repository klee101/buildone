"""Embed the accepted visual meshes and their sRGB colors in an offline viewer."""
import base64
import json
from pathlib import Path

import numpy as np


def encoded(array, dtype):
    return base64.b64encode(array.astype(dtype, copy=False).tobytes()).decode("ascii")


def export_viewer(cache_path, path):
    parts = []
    with np.load(cache_path, allow_pickle=False) as cache:
        visual = json.loads(cache["metadata"].item())["visual"]
        for part in visual.values():
            key = part["mesh"]
            points, faces = cache[key + "_p"], cache[key + "_f"]
            triangles = points[faces]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0])
            normals = np.zeros_like(points)
            for corner in faces.T:
                np.add.at(normals, corner, face_normals)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
            parts.append(dict(vertices=encoded(np.column_stack((points, normals)), "<f4"),
                              indices=encoded(faces, "<u4"), rgb=part["rgb"],
                              low=points.min(axis=0).tolist(), high=points.max(axis=0).tolist()))
    template = Path(__file__).with_name("viewer.html").read_text()
    temp = path.with_suffix(".partial.html")
    temp.write_text(template.replace("/*MESHES*/[]", json.dumps(parts, separators=(",", ":"))))
    temp.replace(path)
