"""
preprocess/volume_preprocess.py
--------------------------------
Preprocessing pipeline for 3D CT volumes (DICOM / NIfTI).

Steps:
    1. Load DICOM series or NIfTI files
    2. Resample to isotropic 1mm³ spacing
    3. Clip HU values to [-1000, 400]
    4. MinMax normalization to [0, 1]
    5. Resize to target shape (64×128×128)
    6. Data augmentation (train only)
    7. Export as .npy for fast loading

Usage:
    python src/preprocess/volume_preprocess.py --input data/raw/scans --output data/processed/volumes
"""

import numpy as np
import SimpleITK as sitk
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional


# ─── Constants ────────────────────────────────────────────────────────────────

HU_MIN = -1000   # Air
HU_MAX = 400     # Soft tissue / bone boundary
TARGET_SHAPE = (64, 128, 128)   # (D, H, W)
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm³ isotropic


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Load a DICOM series from a directory into a SimpleITK Image.

    Args:
        dicom_dir: Path to directory containing .dcm files.

    Returns:
        SimpleITK Image with correct spacing and orientation.
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def load_nifti(nifti_path: str) -> sitk.Image:
    """Load a NIfTI (.nii / .nii.gz) file."""
    return sitk.ReadImage(nifti_path)


def load_volume(path: str) -> sitk.Image:
    """
    Auto-detect format and load volume.

    Args:
        path: Path to DICOM directory or NIfTI file.
    """
    p = Path(path)
    if p.is_dir():
        return load_dicom_series(str(p))
    elif p.suffix in [".nii", ".gz"]:
        return load_nifti(str(p))
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_volume(
    image: sitk.Image,
    target_spacing: tuple = TARGET_SPACING,
    interpolator=sitk.sitkLinear
) -> sitk.Image:
    """
    Resample a CT volume to a target isotropic voxel spacing.

    Args:
        image:          Input SimpleITK Image.
        target_spacing: Desired spacing in mm (D, H, W).
        interpolator:   SimpleITK interpolation method.

    Returns:
        Resampled SimpleITK Image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


# ─── Normalization ────────────────────────────────────────────────────────────

def clip_and_normalize(
    volume: np.ndarray,
    hu_min: float = HU_MIN,
    hu_max: float = HU_MAX
) -> np.ndarray:
    """
    Clip Hounsfield Units and normalize to [0, 1].

    Args:
        volume: 3D numpy array of HU values.
        hu_min: Lower clip bound (air = -1000).
        hu_max: Upper clip bound (soft tissue boundary = 400).

    Returns:
        Normalized float32 array in [0, 1].
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


# ─── Resize ───────────────────────────────────────────────────────────────────

def resize_volume(
    volume: np.ndarray,
    target_shape: tuple = TARGET_SHAPE
) -> np.ndarray:
    """
    Resize a 3D volume to a target shape using SimpleITK.

    Args:
        volume:       Input numpy array (D, H, W).
        target_shape: Desired output shape (D, H, W).

    Returns:
        Resized float32 array.
    """
    sitk_image = sitk.GetImageFromArray(volume)
    original_size = sitk_image.GetSize()         # (W, H, D) in SimpleITK
    original_spacing = sitk_image.GetSpacing()

    # Target size in SimpleITK order (W, H, D)
    target_sitk_size = [target_shape[2], target_shape[1], target_shape[0]]

    new_spacing = [
        original_spacing[i] * original_size[i] / target_sitk_size[i]
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_sitk_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(sitk_image)

    return sitk.GetArrayFromImage(resampled).astype(np.float32)


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_volume(volume: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply random augmentations to a 3D CT volume.

    Augmentations (all random):
    - Horizontal flip (axial plane)
    - Rotation ±15°
    - Gaussian noise
    - Intensity shift ±0.05

    Args:
        volume: (D, H, W) float32 array, normalized to [0, 1].
        seed:   Optional random seed.

    Returns:
        Augmented volume.
    """
    rng = np.random.RandomState(seed)

    # Random flip (left-right)
    if rng.rand() > 0.5:
        volume = np.flip(volume, axis=2).copy()

    # Random flip (anterior-posterior)
    if rng.rand() > 0.5:
        volume = np.flip(volume, axis=1).copy()

    # Random Gaussian noise
    if rng.rand() > 0.5:
        noise_std = rng.uniform(0.005, 0.02)
        volume = np.clip(volume + rng.normal(0, noise_std, volume.shape), 0, 1)

    # Random intensity shift
    if rng.rand() > 0.5:
        shift = rng.uniform(-0.05, 0.05)
        volume = np.clip(volume + shift, 0, 1)

    return volume.astype(np.float32)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def preprocess_volume(
    path: str,
    target_shape: tuple = TARGET_SHAPE,
    augment: bool = False
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single CT scan.

    Args:
        path:         Path to DICOM dir or NIfTI file.
        target_shape: Output volume shape (D, H, W).
        augment:      Apply random augmentation.

    Returns:
        Preprocessed float32 volume of shape target_shape.
    """
    # 1. Load
    image = load_volume(path)

    # 2. Resample to isotropic spacing
    image = resample_volume(image, TARGET_SPACING)

    # 3. Convert to numpy (HU values)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    # 4. Clip + normalize
    volume = clip_and_normalize(volume)

    # 5. Resize to target shape
    volume = resize_volume(volume, target_shape)

    # 6. Augment (train only)
    if augment:
        volume = augment_volume(volume)

    return volume


# ─── Batch Processing ─────────────────────────────────────────────────────────

def process_dataset(
    input_dir: str,
    output_dir: str,
    augment: bool = False,
    file_ext: str = ".nii.gz"
) -> None:
    """
    Preprocess all CT scans in a directory and save as .npy files.

    Args:
        input_dir:  Root directory with scan files/folders.
        output_dir: Directory to save preprocessed .npy volumes.
        augment:    Apply augmentation (use for train split only).
        file_ext:   File extension to search for ('.nii.gz' or '' for DICOM dirs).
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)

    if file_ext:
        paths = list(input_path.rglob(f"*{file_ext}"))
    else:
        # DICOM: each subdirectory is one series
        paths = [p for p in input_path.iterdir() if p.is_dir()]

    print(f"Processing {len(paths)} volumes → {output_dir}")
    errors = []

    for p in tqdm(paths, desc="Preprocessing"):
        try:
            volume = preprocess_volume(str(p), augment=augment)
            out_name = p.stem.replace(".nii", "") + ".npy"
            np.save(os.path.join(output_dir, out_name), volume)
        except Exception as e:
            errors.append((str(p), str(e)))

    print(f"Done. {len(paths) - len(errors)}/{len(paths)} processed.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT volume preprocessor")
    parser.add_argument("--input", type=str, default="data/raw/scans")
    parser.add_argument("--output", type=str, default="data/processed/volumes")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--ext", type=str, default=".nii.gz")
    args = parser.parse_args()

    process_dataset(args.input, args.output, augment=args.augment, file_ext=args.ext)
    print("\nNext step: run text_preprocess.py")
