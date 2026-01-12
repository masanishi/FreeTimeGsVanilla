"""
Normalization utilities for FreeTime dataset.

These functions handle coordinate normalization for camera poses and 3D points:
1. Center the scene at the mean camera position
2. Scale based on camera distribution
3. Align points to principal axes

This is critical for stable training of Gaussian Splatting models.
"""

import numpy as np
from typing import Tuple


def similarity_from_cameras(
    camtoworlds: np.ndarray,
    strict_scaling: bool = False,
) -> np.ndarray:
    """
    Compute a similarity transform that centers and scales the scene
    based on camera positions.

    The transform:
    1. Translates so mean camera position is at origin
    2. Scales so cameras fit within a reasonable range

    Args:
        camtoworlds: [N, 4, 4] camera-to-world matrices
        strict_scaling: If True, use strict unit scaling. Otherwise use
                       distance-based scaling for better numerical stability.

    Returns:
        [4, 4] similarity transform matrix T such that
        T @ [camera_position; 1] gives normalized position
    """
    # Extract camera centers (translation part of cam-to-world)
    camera_centers = camtoworlds[:, :3, 3]  # [N, 3]

    # Compute center (mean position)
    center = np.mean(camera_centers, axis=0)  # [3]

    # Compute scale based on max distance from center
    centered = camera_centers - center
    dists = np.linalg.norm(centered, axis=1)
    max_dist = np.max(dists)

    # Scale factor: want cameras to be roughly within unit sphere
    # Use max_dist to normalize
    if max_dist > 1e-6:
        scale = 1.0 / max_dist
    else:
        scale = 1.0

    if strict_scaling:
        # Strict unit scaling
        scale = 1.0

    # Build similarity transform:
    # First translate to origin, then scale
    # T = S @ Tr where Tr translates and S scales
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] *= scale
    T[:3, 3] = -scale * center

    return T


def transform_cameras(T: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """
    Apply a similarity transformation to camera-to-world matrices.

    For a similarity transform T = [sR | t], we need to:
    1. Transform camera positions: p' = sR @ p + t
    2. Transform camera rotations: R' = R_T @ R (rotation only, no scale!)

    This ensures the output cam-to-world matrices remain valid (orthonormal rotation).

    Args:
        T: [4, 4] similarity transformation matrix (may include scale)
        camtoworlds: [N, 4, 4] camera-to-world matrices

    Returns:
        [N, 4, 4] transformed camera-to-world matrices with valid rotations
    """
    N = len(camtoworlds)
    transformed = np.zeros_like(camtoworlds)

    # Extract rotation and scale from T
    T_rot_scale = T[:3, :3]
    T_trans = T[:3, 3]

    # Decompose T_rot_scale into scale and rotation using SVD
    U, S, Vt = np.linalg.svd(T_rot_scale)
    scale = S[0]  # Assume uniform scale (S should be [s, s, s])
    R_T = U @ Vt  # Pure rotation

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R_T) < 0:
        R_T = -R_T

    for i in range(N):
        c2w = camtoworlds[i]
        R_cam = c2w[:3, :3]  # Camera rotation
        t_cam = c2w[:3, 3]   # Camera position

        # Transform rotation: R' = R_T @ R (keeps orthonormality)
        R_new = R_T @ R_cam

        # Transform position: t' = scale * R_T @ t + T_trans
        t_new = scale * (R_T @ t_cam) + T_trans

        # Build new c2w
        transformed[i, :3, :3] = R_new
        transformed[i, :3, 3] = t_new
        transformed[i, 3, 3] = 1.0

    return transformed.astype(np.float32)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a transformation to 3D points.

    Args:
        T: [4, 4] transformation matrix
        points: [N, 3] 3D points

    Returns:
        [N, 3] transformed points
    """
    if len(points) == 0:
        return points

    # Convert to homogeneous coordinates
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)  # [N, 4]

    # Apply transform
    transformed_h = (T @ points_h.T).T  # [N, 4]

    # Convert back to 3D
    return transformed_h[:, :3].astype(np.float32)


def align_principle_axes(points: np.ndarray) -> np.ndarray:
    """
    Compute a rotation that aligns the point cloud to its principal axes.

    This uses PCA to find the principal directions of the point cloud,
    then rotates so these align with the coordinate axes.

    The rotation is chosen to:
    1. Align the largest variance direction with Z axis
    2. Align the second largest with Y axis
    3. Smallest variance with X axis

    This helps with consistent visualization and can improve training stability.

    Args:
        points: [N, 3] 3D points

    Returns:
        [4, 4] rotation matrix (pure rotation, no translation/scale)
    """
    if len(points) < 4:
        # Not enough points for meaningful PCA
        return np.eye(4, dtype=np.float32)

    # Center points for PCA
    center = np.mean(points, axis=0)
    centered = points - center

    # Compute covariance matrix
    cov = np.cov(centered.T)  # [3, 3]

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest last for eigh)
    # We want largest first
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # Build rotation matrix from eigenvectors
    # eigenvectors are column vectors
    R = eigenvectors.T  # [3, 3]

    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(R) < 0:
        R[2, :] *= -1  # Flip one axis

    # Build 4x4 transform (rotation only, applied around origin)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R

    return T


def compute_scene_extent(
    camtoworlds: np.ndarray,
    points: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute scene center and extent from cameras and optionally points.

    Args:
        camtoworlds: [N, 4, 4] camera-to-world matrices
        points: Optional [M, 3] 3D points

    Returns:
        center: [3] scene center
        extent: scalar scene extent (radius)
    """
    # Camera centers
    camera_centers = camtoworlds[:, :3, 3]

    if points is not None and len(points) > 0:
        all_points = np.concatenate([camera_centers, points], axis=0)
    else:
        all_points = camera_centers

    center = np.mean(all_points, axis=0)
    dists = np.linalg.norm(all_points - center, axis=1)
    extent = np.max(dists)

    return center.astype(np.float32), float(extent)


def normalize_scene(
    camtoworlds: np.ndarray,
    points: np.ndarray,
    target_extent: float = 1.0,
    align_axes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully normalize a scene: center, scale, and optionally align to principal axes.

    Args:
        camtoworlds: [N, 4, 4] camera-to-world matrices
        points: [M, 3] 3D points
        target_extent: Target scene extent after normalization
        align_axes: Whether to align to principal axes

    Returns:
        camtoworlds: Normalized camera poses
        points: Normalized 3D points
        transform: [4, 4] full transformation matrix applied
    """
    # Step 1: Similarity transform (center and scale)
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)

    # Step 2: Align to principal axes
    if align_axes:
        T2 = align_principle_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        transform = T2 @ T1
    else:
        transform = T1

    # Step 3: Additional scaling to target extent
    _, current_extent = compute_scene_extent(camtoworlds, points)
    if current_extent > 1e-6:
        scale = target_extent / current_extent
        T3 = np.eye(4, dtype=np.float32)
        T3[:3, :3] *= scale
        camtoworlds = transform_cameras(T3, camtoworlds)
        points = transform_points(T3, points)
        transform = T3 @ transform

    return camtoworlds, points, transform
