"""Build each CAD version once, cache triangles and rasterize its four-view feedback."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from reference import VIEWS


def rasterize(points, faces, colors, folder):
    # Right, up, towards-camera axes for front, top, left and right, respectively.
    bases = [((0, 1), (2, 1), (1, -1)), ((0, 1), (1, 1), (2, 1)),
             ((1, -1), (2, 1), (0, -1)), ((1, 1), (2, 1), (0, 1))]
    size = 360
    sheet = Image.new("RGB", (2*size, 2*(size+22)), "white")
    scale = (size-30) / np.ptp(points, axis=0).max()
    center = (points.min(axis=0)+points.max(axis=0)) / 2
    triangles = points[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    for i, basis in enumerate(bases):
        axes, signs = zip(*basis)
        p = (points-center)[:, axes] * signs
        p[:, :2] = p[:, :2] * (scale, -scale) + size/2
        light = np.array([-.34, .52, .78])
        lambert = np.clip((normals[:, axes] * signs) @ light, 0, 1)
        shades = np.uint8(np.clip(colors*(.52 + .48*lambert[:, None])*255, 0, 255))
        pixels = np.full((size, size, 3), 241, dtype=np.uint8)
        depth = np.full((size, size), -np.inf)
        for tri, color in zip(p[faces], shades):
            a, b, c = tri
            x0, y0 = np.maximum(0, np.floor(tri[:, :2].min(axis=0)).astype(int))
            x1, y1 = np.minimum(size-1, np.ceil(tri[:, :2].max(axis=0)).astype(int))
            bx, by, bz = b-a
            cx, cy, cz = c-a
            det = bx*cy - by*cx
            if abs(det) < 1e-8:
                continue
            yy, xx = np.ogrid[y0:y1+1, x0:x1+1]
            dx, dy = xx+.5-a[0], yy+.5-a[1]
            u, v = (dx*cy-dy*cx)/det, (bx*dy-by*dx)/det
            z = a[2]+u*bz+v*cz
            tile = np.s_[y0:y1+1, x0:x1+1]
            visible = (u >= 0) & (v >= 0) & (u+v <= 1) & (z > depth[tile])
            depth[tile][visible] = z[visible]
            pixels[tile][visible] = color
        left, top = i % 2 * size, i // 2 * (size+22)
        sheet.paste(Image.fromarray(pixels), (left, top+22))
        ImageDraw.Draw(sheet).text((left+8, top+3), VIEWS[i], fill="black")
    sheet.save(folder / "preview.png")


def build_version(folder, known):
    start = perf_counter()
    model = runpy.run_path(str(folder / "model.py"))
    visual, collision = model["build"]()
    if not visual or not collision:
        raise ValueError("build() must supply both visual and collision parts")
    data, parts = {}, {"visual": {}, "collision": {}}

    def mesh(name, shape):
        verts, faces = shape.tessellate(.12, .15)
        p = np.asarray([tuple(v) for v in verts], dtype=np.float64)
        f = np.asarray(faces, dtype=np.int32)
        if not len(f) or not np.isfinite(p).all():
            raise ValueError(f"Empty or nonfinite geometry for {name}")
        data[name + "_p"], data[name + "_f"] = p, f
        return name

    points, faces, colors = [], [], []
    offset = 0
    for name, (shape, rgb) in visual.items():
        if not name.isascii() or not name.isidentifier():
            raise ValueError(f"Invalid visual name {name!r}")
        rgb = np.array(rgb, dtype=float)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or min(rgb) < 0 or max(rgb) > 1:
            raise ValueError(f"Invalid color for {name}")
        key = mesh("vis_" + name, shape)
        parts["visual"][name] = dict(mesh=key, rgb=rgb.tolist())
        p, f = data[key+"_p"], data[key+"_f"]
        points.append(p)
        faces.append(f+offset)
        colors.append(np.tile(rgb, (len(f), 1)))
        offset += len(p)
    for name, shape in collision.items():
        if not name.isascii() or not name.isidentifier():
            raise ValueError(f"Invalid collider name {name!r}")
        parts["collision"][name] = (parts["visual"][shape]["mesh"] if isinstance(shape, str)
                                    else mesh("col_" + name, shape))

    points = np.concatenate(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    dims = hi-lo
    origin = (hi+lo)/2
    origin[2] = lo[2]
    for key in data:
        if key.endswith("_p"):
            data[key] = np.asarray((data[key]-origin)*.001, dtype=np.float32)
    sizes = dict(zip(("width", "depth", "height"), dims.tolist()))
    mismatches = {k: f"{sizes[k]:.3f} mm (requested {v} mm)" for k, v in known.items()
                  if abs(sizes[k]-v) > max(.05, v*.0005)}
    parts.update(dimensions_mm=sizes, scale_note=str(model["SCALE_NOTE"]),
                 model_sha256=hashlib.sha256((folder/"model.py").read_bytes()).hexdigest())
    np.savez_compressed(folder/"mesh.npz", metadata=json.dumps(parts), **data)
    build_seconds = perf_counter()-start
    rasterize(points, np.concatenate(faces), np.concatenate(colors), folder)
    return dict(usable=not mismatches, dimension_errors=mismatches, **parts,
                build_cache_seconds=build_seconds, preview_seconds=perf_counter()-start-build_seconds)


def modeling(out, target, sam, known, ask, attempt, record):
    evidence = []
    for view in sam["views"]:
        role = view["role"]
        evidence.extend(out / "evidence" / f"{role}-{kind}.png" for kind in ("source", "isolated"))
    evidence.extend(out / "references" / f"{name}.png" for name in VIEWS)
    skill = (Path(__file__).parent / "skills/cad/SKILL.md").read_text()
    prompt = (skill + "\nTarget and front: " + json.dumps(target) +
              "\nSupplied dimensions in mm: " + json.dumps(known) +
              "\nImages, in order: " + ", ".join(p.name for p in evidence) +
              "\nWrite the complete model.py as your response, plain Python source only.")
    source, thread = ask("model", prompt, evidence, suffix=".py")
    accepted = None
    for version in range(2):
        folder = out / "cad" / f"v{version}"
        already_started = folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        (out/"model.py").write_text(source)
        if already_started:
            saved = folder/"cad.json"
            result = (json.loads(saved.read_text()) if saved.is_file() else
                      dict(usable=False, error="CAD process interrupted before producing its result"))
        else:
            shutil.copyfile(out/"model.py", folder/"model.py")
            result = attempt(folder, version)
        record["attempts"].append(result)
        record["build_count"] = len(record["attempts"])
        if result["usable"]:
            accepted = folder, result
        if version == 0:
            feedback = {k: result.get(k) for k in ("usable", "dimensions_mm", "dimension_errors", "error")}
            images = [folder/"preview.png"] if (folder/"preview.png").exists() else []
            prompt = ("Review v0 against the same evidence and CAD skill. Results: " + json.dumps(feedback)
                      + "\nReply exactly KEEP if usable or return the complete edited Python source "
                        "for one targeted correction. Only one more build is available.")
            try:
                source, _ = ask("review", prompt, images, session=thread, suffix=".txt")
            except Exception as exc:
                if not accepted:
                    raise
                record["review_error"] = str(exc)
                break
            if source.strip() == "KEEP":
                break
    if not accepted:
        raise RuntimeError(f"No usable CAD build: {record['attempts']}")
    folder, result = accepted
    shutil.copyfile(folder/"model.py", out/"model.py")
    record.update(accepted_version=folder.name, mesh_cache=str((folder/"mesh.npz").relative_to(out)),
                  dimensions_mm=result["dimensions_mm"], scale_note=result["scale_note"])
    return folder/"mesh.npz"
