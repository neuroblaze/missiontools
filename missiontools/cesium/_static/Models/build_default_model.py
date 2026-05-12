#!/usr/bin/env -S uv run python3
"""Build the default spacecraft model: textured cube with face labels.

Creates a 1m³ cube with UV-mapped faces labelled "X+", "X−", "Y+", "Y−",
"Z+", "Z−" in the corresponding regions of a 3×2 atlas texture.

This script is intentionally standalone — it needs no missiontools imports.
Run from repo root::

    python3 missiontools/cesium/_static/Models/build_default_model.py
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLB_PATH = os.path.join(_SCRIPT_DIR, "..", "Cesium", "Models", "default_spacecraft.glb")
TEX_SIZE = 512  # atlas texture resolution (pixels)

# ---------------------------------------------------------------------------
# Texture atlas layout (3 cols × 2 rows)
# ---------------------------------------------------------------------------
#  | X+ | X− | Y+ |
#  | Y− | Z+ | Z− |
# ---------------------------------------------------------------------------

_ATLAS = {
    "+X": (0, 1),  # col 0, row 1 (top)
    "-X": (1, 1),
    "+Y": (2, 1),
    "-Y": (0, 0),  # col 0, row 0 (bottom)
    "+Z": (1, 0),
    "-Z": (2, 0),
}

# Labels drawn onto the atlas for each glTF face.
# Cesium auto-converts glTF Y-up → Z-up with Rx(−90°), remapping:
#   glTF +Y → Cesium +Z     glTF +Z → Cesium −Y
#   glTF +X → Cesium +X (unchanged)
# We label each *glTF* face with the Cesium-axis text that will be
# visible after Cesium's automatic conversion.
_LABEL = {
    "+X": "X+",
    "-X": "X−",
    "+Y": "Z+",
    "-Y": "Z−",
    "+Z": "Y−",
    "-Z": "Y+",
}


# ---------------------------------------------------------------------------
# 1. Build texture image
# ---------------------------------------------------------------------------


def _build_texture() -> Image.Image:
    img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            size=TEX_SIZE // 12,
        )
    except OSError:
        font = ImageFont.load_default()

    cols, rows = 3, 2
    cw = TEX_SIZE // cols
    rh = TEX_SIZE // rows

    # Fill each cell with a distinct dark colour and draw label
    palette = [
        (180, 50, 50),  # X+ dark red
        (50, 180, 50),  # X− dark green
        (50, 50, 180),  # Y+ dark blue
        (180, 160, 50),  # Y− dark gold
        (180, 80, 20),  # Z+ dark orange
        (90, 60, 160),  # Z− dark purple
    ]
    idx = 0

    for face_key in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        col, row = _ATLAS[face_key]
        x0, y0 = col * cw, (1 - row) * rh  # PIL y=0 is top
        x1, y1 = x0 + cw, y0 + rh
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=palette[idx])
        text = _LABEL[face_key]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + (cw - tw) // 2
        ty = y0 + (rh - th) // 2
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
        idx += 1

    return img


# ---------------------------------------------------------------------------
# 2. Build cube mesh with UVs
# ---------------------------------------------------------------------------


def _cube_face(
    corners: list[tuple[float, float, float]],
    face_key: str,
    cols: int = 3,
    rows: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices 4×3, faces 2×3, uvs 4×2) for one quad.

    *corners* are ccw when viewed from outside.
    """
    v = np.array(corners, dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    col, row = _ATLAS[face_key]
    u0, u1 = col / cols, (col + 1) / cols
    v0, v1 = row / rows, (row + 1) / rows
    uv = np.array(
        [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],  # ccw, matching corners
        dtype=np.float64,
    )
    return v, f, uv


def _build_cube_mesh(margin: float = 0.0) -> trimesh.Trimesh:
    """Build a 1×1×1 cube centred at origin with per-face UVs.

    Each face's vertices are pushed outward 1 μm along its normal so
    all 24 vertices remain distinct — trimesh would otherwise deduplicate
    coincident vertices and discard the per-face UV data.
    """
    r = 0.5 - margin
    epsilon = 1e-6  # prevents vertex dedup, invisible at any scale

    faces_def: list[tuple[str, list[tuple[float, float, float]]]] = [
        (
            "+X",
            [
                (r + epsilon, r, -r),
                (r + epsilon, r, r),
                (r + epsilon, -r, r),
                (r + epsilon, -r, -r),
            ],
        ),
        (
            "-X",
            [
                (-r - epsilon, r, r),
                (-r - epsilon, r, -r),
                (-r - epsilon, -r, -r),
                (-r - epsilon, -r, r),
            ],
        ),
        (
            "+Y",
            [
                (r, r + epsilon, r),
                (r, r + epsilon, -r),
                (-r, r + epsilon, -r),
                (-r, r + epsilon, r),
            ],
        ),
        (
            "-Y",
            [
                (r, -r - epsilon, -r),
                (r, -r - epsilon, r),
                (-r, -r - epsilon, r),
                (-r, -r - epsilon, -r),
            ],
        ),
        (
            "+Z",
            [
                (-r, -r, r + epsilon),
                (r, -r, r + epsilon),
                (r, r, r + epsilon),
                (-r, r, r + epsilon),
            ],
        ),
        (
            "-Z",
            [
                (-r, r, -r - epsilon),
                (r, r, -r - epsilon),
                (r, -r, -r - epsilon),
                (-r, -r, -r - epsilon),
            ],
        ),
    ]

    verts_list, faces_list, uv_list = [], [], []
    v_offset = 0
    for key, corners in faces_def:
        v, f, uv = _cube_face(corners, key)
        verts_list.append(v)
        faces_list.append(f + v_offset)
        uv_list.append(uv)
        v_offset += 4

    all_verts = np.vstack(verts_list)
    all_faces = np.vstack(faces_list)
    all_uvs = np.vstack(uv_list)

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces)
    mesh.visual = trimesh.visual.TextureVisuals(uv=all_uvs)
    return mesh


# ---------------------------------------------------------------------------
# 3. Main — compose and export
# ---------------------------------------------------------------------------


def main() -> None:
    img = _build_texture()

    mesh = _build_cube_mesh(margin=0.0)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        material=material,
    )

    mesh.export(GLB_PATH)
    print(f"  Wrote {GLB_PATH}")


if __name__ == "__main__":
    main()
