"""PyRadiomics feature extraction for raw and dentin-normalized CBCT images."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as connected_components

try:
    import SimpleITK as sitk
    from radiomics import featureextractor
except ImportError:  # pragma: no cover
    sitk = None
    featureextractor = None


DIAGNOSTIC_PREFIX = "diagnostics_"


def _require_radiomics():
    if sitk is None or featureextractor is None:
        raise ImportError("SimpleITK and PyRadiomics are required. Install SimpleITK and pyradiomics.")


def read_sitk(path: str | Path):
    """Read an image using SimpleITK to preserve spacing and orientation metadata."""
    _require_radiomics()
    return sitk.ReadImage(str(path))


def largest_component_binary_mask(label_image, lesion_label: int = 1, connectivity: int = 1):
    """Return a SimpleITK binary mask for the largest connected lesion component."""
    _require_radiomics()
    label_arr = sitk.GetArrayFromImage(label_image)
    lesion = label_arr == lesion_label
    if not np.any(lesion):
        raise ValueError(f"No voxels found for lesion_label={lesion_label}.")

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    cc, n_cc = connected_components(lesion, structure=structure)
    if n_cc == 0:
        raise ValueError("No connected lesion component found.")

    component_sizes = np.bincount(cc.ravel())
    component_sizes[0] = 0
    largest_id = int(np.argmax(component_sizes))
    mask_arr = (cc == largest_id).astype(np.uint8)

    mask_image = sitk.GetImageFromArray(mask_arr)
    mask_image.CopyInformation(label_image)
    return mask_image, int(component_sizes[largest_id]), int(n_cc)


def scalarize_feature_value(value):
    """Convert PyRadiomics output values into CSV-safe scalars or strings."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.ravel()[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def clean_pyradiomics_result(result: Mapping, drop_diagnostics: bool = True) -> dict:
    """Remove diagnostics and convert values into serializable scalars."""
    out = {}
    for key, value in result.items():
        if drop_diagnostics and str(key).startswith(DIAGNOSTIC_PREFIX):
            continue
        out[str(key)] = scalarize_feature_value(value)
    return out


def make_extractor(config_path: str | Path | None = None, bin_width: float | None = None):
    """Create a PyRadiomics extractor from YAML or explicit bin width."""
    _require_radiomics()
    if config_path is not None:
        return featureextractor.RadiomicsFeatureExtractor(str(config_path))
    extractor = featureextractor.RadiomicsFeatureExtractor()
    if bin_width is not None:
        extractor.settings.update({"binWidth": bin_width})
    return extractor


def extract_features_for_case(
    image_path: str | Path,
    label_path: str | Path,
    case_id: str,
    extractor,
    lesion_label: int = 1,
    connectivity: int = 1,
    drop_diagnostics: bool = True,
) -> dict:
    """Extract PyRadiomics features from the largest lesion component of one case."""
    image = read_sitk(image_path)
    label = read_sitk(label_path)
    mask, lesion_voxels, n_components = largest_component_binary_mask(
        label, lesion_label=lesion_label, connectivity=connectivity
    )
    result = extractor.execute(image, mask)
    features = clean_pyradiomics_result(result, drop_diagnostics=drop_diagnostics)
    features.update({
        "case_id": case_id,
        "lesion_voxels_largest_component": lesion_voxels,
        "lesion_component_count": n_components,
    })
    return features


def extract_features_from_manifest(
    manifest: pd.DataFrame,
    image_column: str,
    output_csv: str | Path,
    label_column: str = "label_path",
    case_id_column: str = "case_id",
    config_path: str | Path | None = None,
    bin_width: float | None = 100,
    lesion_label: int = 1,
    connectivity: int = 1,
    metadata_columns: Sequence[str] = ("diagnosis",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features from all rows in a manifest and save a feature table.

    Metadata columns such as diagnosis are copied into the output feature table
    when present. They are not used during feature extraction, but keeping them
    avoids fragile label reconstruction from case-id suffixes during ML analysis.
    """
    extractor = make_extractor(config_path=config_path, bin_width=bin_width)
    records = []
    failures = []

    for _, row in manifest.iterrows():
        case_id = str(row[case_id_column]) if case_id_column in row else Path(str(row[image_column])).stem
        try:
            rec = extract_features_for_case(
                image_path=row[image_column],
                label_path=row[label_column],
                case_id=case_id,
                extractor=extractor,
                lesion_label=lesion_label,
                connectivity=connectivity,
            )
            for meta_col in metadata_columns:
                if meta_col in manifest.columns:
                    rec[meta_col] = row[meta_col]
            records.append(rec)
        except Exception as exc:  # pragma: no cover
            failures.append({"case_id": case_id, "image_path": row[image_column], "error": str(exc)})

    features_df = pd.DataFrame(records)
    if not features_df.empty:
        features_df = features_df.set_index("case_id")
        features_df.to_csv(output_csv)

    failure_df = pd.DataFrame(failures)
    return features_df, failure_df
