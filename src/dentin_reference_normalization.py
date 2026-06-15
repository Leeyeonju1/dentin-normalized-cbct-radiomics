"""Dentin reference ROI extraction and affine intensity normalization for CBCT volumes.

Expected segmentation labels:
    1 = lesion
    2 = radiopaque material or restoration
    3 = bone
    4 = tooth structure

The selected reference region is a local tooth-root reference ROI adjacent to the
lesion-facing root surface. If the segmentation label does not distinguish enamel
from dentin, this region should be described as a dentin-proxy reference ROI.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as connected_components
from scipy.stats import trim_mean

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


@dataclass
class DentinRoiConfig:
    tooth_label: int = 4
    lesion_label: int = 1
    bone_label: int = 3
    material_label: int = 2
    margin_voxels: int = 5
    local_radius_voxels: int = 35
    material_buffer_voxels: int = 2
    min_voxels: int = 100
    clip_lower_percentile: float = 1.0
    clip_upper_percentile: float = 99.0
    connectivity: int = 1


@dataclass
class DentinStats:
    source_mu: float
    source_sigma: float
    n_voxels: int
    p_low: float
    p_high: float


def _require_nibabel():
    if nib is None:
        raise ImportError("nibabel is required for NIfTI input and output. Install nibabel first.")


def load_nifti_array(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as a floating point NumPy array."""
    _require_nibabel()
    return nib.load(str(path)).get_fdata()


def save_nifti_like(data: np.ndarray, output_path: str | Path, reference_path: str | Path) -> None:
    """Save a NIfTI volume while preserving affine and header from a reference image."""
    _require_nibabel()
    ref = nib.load(str(reference_path))
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=ref.affine, header=header)
    nib.save(out, str(output_path))


def make_normalized_path(image_path: str | Path, suffix: str = "_dentin_normalized") -> Path:
    """Create an output path by appending a suffix before the NIfTI extension."""
    image_path = Path(image_path)
    name = image_path.name
    if name.endswith(".nii.gz"):
        return image_path.with_name(f"{name[:-7]}{suffix}.nii.gz")
    if name.endswith(".nii"):
        return image_path.with_name(f"{name[:-4]}{suffix}.nii")
    return image_path.with_name(f"{name}{suffix}.nii.gz")


def extract_dentin_reference_roi(
    image: np.ndarray,
    label: np.ndarray,
    config: DentinRoiConfig = DentinRoiConfig(),
) -> tuple[np.ndarray, DentinStats, dict[str, np.ndarray]]:
    """Extract a local dentin reference ROI and compute clipped ROI statistics.

    The method identifies the tooth component adjacent to the lesion, restricts
    the candidate region to the local root neighborhood, selects tooth voxels
    close to the root-bone boundary, excludes voxels near radiopaque material,
    clips ROI intensities to reduce extreme artifacts, and computes mean and SD.
    """
    image = np.asarray(image)
    label = np.asarray(label)

    if image.shape != label.shape:
        raise ValueError(f"Image and label shapes differ: {image.shape} vs {label.shape}")

    structure = generate_binary_structure(rank=3, connectivity=config.connectivity)

    tooth_mask = label == config.tooth_label
    lesion_mask = label == config.lesion_label
    bone_mask = label == config.bone_label
    material_mask = label == config.material_label

    if not np.any(tooth_mask):
        raise ValueError("No tooth-label voxels were found.")
    if not np.any(lesion_mask):
        raise ValueError("No lesion-label voxels were found.")
    if not np.any(bone_mask):
        raise ValueError("No bone-label voxels were found.")

    tooth_cc, n_cc = connected_components(tooth_mask, structure=structure)
    lesion_dilated = binary_dilation(lesion_mask, structure=structure)
    adjacent_ids = np.unique(tooth_cc[lesion_dilated & tooth_mask])
    adjacent_ids = adjacent_ids[adjacent_ids > 0]

    if len(adjacent_ids) == 0:
        dist_to_lesion = distance_transform_edt(~lesion_mask)
        nearest_id = None
        nearest_dist = np.inf
        for component_id in range(1, n_cc + 1):
            component = tooth_cc == component_id
            if not np.any(component):
                continue
            dist = float(np.min(dist_to_lesion[component]))
            if dist < nearest_dist:
                nearest_id = component_id
                nearest_dist = dist
        if nearest_id is None:
            raise ValueError("No valid tooth component was found.")
        adjacent_ids = np.array([nearest_id])

    lesion_tooth_mask = np.isin(tooth_cc, adjacent_ids)
    dist_to_lesion = distance_transform_edt(~lesion_mask)
    local_tooth_mask = lesion_tooth_mask & (dist_to_lesion <= config.local_radius_voxels)

    bone_dilated = binary_dilation(bone_mask, structure=structure)
    root_bone_boundary = local_tooth_mask & bone_dilated
    if not np.any(root_bone_boundary):
        raise ValueError("No local root-bone boundary was found.")

    dist_to_boundary = distance_transform_edt(~root_bone_boundary)
    roi_mask = (
        local_tooth_mask
        & (dist_to_boundary > 0)
        & (dist_to_boundary <= config.margin_voxels)
    )

    if np.any(material_mask):
        dist_to_material = distance_transform_edt(~material_mask)
        roi_mask &= dist_to_material > config.material_buffer_voxels

    roi_values = image[roi_mask]
    if roi_values.size < config.min_voxels:
        raise ValueError(f"Reference ROI too small: {roi_values.size} voxels")

    p_low, p_high = np.percentile(
        roi_values,
        [config.clip_lower_percentile, config.clip_upper_percentile],
    )
    clipped = np.clip(roi_values, p_low, p_high)
    source_sigma = float(np.std(clipped))
    if source_sigma <= 1e-8:
        raise ValueError(f"Reference ROI SD too small: {source_sigma}")

    stats = DentinStats(
        source_mu=float(np.mean(clipped)),
        source_sigma=source_sigma,
        n_voxels=int(roi_values.size),
        p_low=float(p_low),
        p_high=float(p_high),
    )

    masks = {
        "roi_mask": roi_mask,
        "local_tooth_mask": local_tooth_mask,
        "root_bone_boundary": root_bone_boundary,
    }
    return roi_mask, stats, masks


def estimate_reference_distribution(
    manifest: pd.DataFrame,
    config: DentinRoiConfig = DentinRoiConfig(),
    trim_proportion: float = 0.10,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Estimate global reference mean and SD from a predefined reference cohort.

    The manifest must contain image_path and label_path columns. It should be
    predefined before model evaluation and should not be selected by outcome.
    """
    records = []
    failures = []

    for _, row in manifest.iterrows():
        case_id = row.get("case_id", Path(str(row["image_path"])).stem)
        try:
            image = load_nifti_array(row["image_path"])
            label = load_nifti_array(row["label_path"])
            _, stats, _ = extract_dentin_reference_roi(image, label, config)
            rec = {"case_id": case_id, **asdict(stats)}
            records.append(rec)
        except Exception as exc:  # pragma: no cover
            failures.append({"case_id": case_id, "error": str(exc)})

    stats_df = pd.DataFrame(records)
    failure_df = pd.DataFrame(failures)
    if stats_df.empty:
        raise RuntimeError("No valid reference ROI was extracted from the reference cohort.")

    reference = {
        "mu_ref": float(trim_mean(stats_df["source_mu"].to_numpy(), proportiontocut=trim_proportion)),
        "sigma_ref": float(trim_mean(stats_df["source_sigma"].to_numpy(), proportiontocut=trim_proportion)),
        "n_reference_images": int(len(stats_df)),
        "trim_proportion": float(trim_proportion),
    }
    return reference, stats_df, failure_df


def apply_affine_dentin_normalization(
    image: np.ndarray,
    source_mu: float,
    source_sigma: float,
    mu_ref: float,
    sigma_ref: float,
) -> tuple[np.ndarray, float, float]:
    """Apply I_norm = a * I_source + b with a = sigma_ref / sigma_source."""
    if source_sigma <= 1e-8:
        raise ValueError("source_sigma must be positive.")
    a = float(sigma_ref / source_sigma)
    b = float(mu_ref - a * source_mu)
    return a * image + b, a, b


def normalize_manifest(
    manifest: pd.DataFrame,
    reference: dict,
    output_dir: str | Path,
    config: DentinRoiConfig = DentinRoiConfig(),
    suffix: str = "_dentin_normalized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize all images in a manifest and save per-case dentin statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []

    for _, row in manifest.iterrows():
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        case_id = row.get("case_id", image_path.stem)
        try:
            image = load_nifti_array(image_path)
            label = load_nifti_array(label_path)
            roi_mask, stats, _ = extract_dentin_reference_roi(image, label, config)

            normalized, a, b = apply_affine_dentin_normalization(
                image=image,
                source_mu=stats.source_mu,
                source_sigma=stats.source_sigma,
                mu_ref=reference["mu_ref"],
                sigma_ref=reference["sigma_ref"],
            )

            output_path = output_dir / make_normalized_path(image_path, suffix=suffix).name
            save_nifti_like(normalized, output_path, image_path)

            clipped_before = np.clip(image[roi_mask], stats.p_low, stats.p_high)
            clipped_after = a * clipped_before + b

            records.append({
                "case_id": case_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "normalized_image_path": str(output_path),
                **asdict(stats),
                "mu_ref": float(reference["mu_ref"]),
                "sigma_ref": float(reference["sigma_ref"]),
                "a": float(a),
                "b": float(b),
                "after_roi_mu": float(np.mean(clipped_after)),
                "after_roi_sigma": float(np.std(clipped_after)),
            })
        except Exception as exc:  # pragma: no cover
            failures.append({"case_id": case_id, "image_path": str(image_path), "error": str(exc)})

    return pd.DataFrame(records), pd.DataFrame(failures)
