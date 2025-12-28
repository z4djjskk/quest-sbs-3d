from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics


def save_ply_fast(
    gaussians: Gaussians3D, f_px: float, image_shape: tuple[int, int], path: Path
) -> None:
    """Save a predicted Gaussian3D to a ply file without Python object explosion."""

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    mean_vectors = gaussians.mean_vectors
    singular_values = gaussians.singular_values
    quaternions = gaussians.quaternions
    color_space_index = cs_utils.encode_color_space("sRGB")

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = int(mean_vectors.reshape(-1, 3).shape[0])

    image_height, image_width = image_shape

    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = ("image_size", image_size_array)

    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = ("intrinsic", intrinsic_array)

    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = ("extrinsic", extrinsic_array)

    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = ("frame", frame_array)

    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)

    if num_gaussians > 0:
        disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
        quantiles = (
            torch.quantile(disparity, q=torch.tensor([0.1, 0.9], device=disparity.device))
            .float()
            .cpu()
            .numpy()
        )
        disparity_array[:] = quantiles
    else:
        disparity_array[:] = np.array([0.0, 0.0], dtype=np.float32)
    disparity_element = ("disparity", disparity_array)

    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = ("color_space", color_space_array)

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = ("version", version_array)

    _write_ply_streaming(
        path,
        num_gaussians,
        dtype_full,
        mean_vectors,
        singular_values,
        quaternions,
        gaussians.colors,
        gaussians.opacities,
        [
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ],
        _inverse_sigmoid,
    )
    return None


def _write_ply_streaming(
    path: Path,
    num_vertices: int,
    vertex_dtype: list[tuple[str, str]],
    mean_vectors: torch.Tensor,
    singular_values: torch.Tensor,
    quaternions: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    tail_elements: list[tuple[str, np.ndarray]],
    inverse_sigmoid,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
    ]

    for name, array in tail_elements:
        header_lines.append(f"element {name} {len(array)}")
        field_dtype = array.dtype
        if array.dtype.fields and name in array.dtype.fields:
            field_dtype = array.dtype.fields[name][0]
        if field_dtype.kind == "f":
            dtype_name = "float"
        elif field_dtype.kind == "u":
            dtype_name = "uchar" if field_dtype.itemsize == 1 else "uint"
        else:
            dtype_name = "int"
        header_lines.append(f"property {dtype_name} {name}")

    header_lines.append("end_header")
    header = "\n".join(header_lines).encode("ascii") + b"\n"

    vertex_dtype_le = [(name, "<f4") for name, _ in vertex_dtype]
    vertex_dtype_np = np.dtype(vertex_dtype_le)

    chunk_size = int(os.environ.get("SHARP_PLY_CHUNK_SIZE", "200000"))
    if chunk_size <= 0:
        chunk_size = num_vertices
    chunk_size = max(1, min(chunk_size, num_vertices))

    with path.open("wb") as f:
        f.write(header)
        if num_vertices <= 0:
            for _, array in tail_elements:
                if array.dtype.byteorder not in ("<", "=", "|"):
                    array = array.byteswap().newbyteorder("<")
                array.tofile(f)
            return
        mean_vectors_flat = mean_vectors.flatten(0, 1)
        quaternions_flat = quaternions.flatten(0, 1)
        singular_values_flat = singular_values.flatten(0, 1)
        colors_flat = colors.flatten(0, 1)
        opacities_flat = opacities.flatten(0, 1)
        for start in range(0, num_vertices, chunk_size):
            end = min(start + chunk_size, num_vertices)
            elements = np.empty(end - start, dtype=vertex_dtype_np)
            xyz_np = (
                mean_vectors_flat[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            scale_np = (
                torch.log(singular_values_flat[start:end])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            quat_np = (
                quaternions_flat[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            colors_slice = colors_flat[start:end]
            colors_np = convert_rgb_to_spherical_harmonics(
                cs_utils.linearRGB2sRGB(colors_slice)
            ).detach().cpu().numpy().astype(np.float32, copy=False)
            opac_slice = opacities_flat[start:end]
            opacity_np = (
                inverse_sigmoid(opac_slice)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            elements["x"] = xyz_np[:, 0]
            elements["y"] = xyz_np[:, 1]
            elements["z"] = xyz_np[:, 2]
            elements["f_dc_0"] = colors_np[:, 0]
            elements["f_dc_1"] = colors_np[:, 1]
            elements["f_dc_2"] = colors_np[:, 2]
            elements["opacity"] = opacity_np
            elements["scale_0"] = scale_np[:, 0]
            elements["scale_1"] = scale_np[:, 1]
            elements["scale_2"] = scale_np[:, 2]
            elements["rot_0"] = quat_np[:, 0]
            elements["rot_1"] = quat_np[:, 1]
            elements["rot_2"] = quat_np[:, 2]
            elements["rot_3"] = quat_np[:, 3]
            elements.tofile(f)

        for _, array in tail_elements:
            if array.dtype.byteorder not in ("<", "=", "|"):
                array = array.byteswap().newbyteorder("<")
            array.tofile(f)
