"""Pose analysis using MediaPipe: keypoint extraction, sprint asymmetry, knee flexion.

Processes video frames or images to extract 33-keypoint skeletal poses,
then computes biomechanical metrics:
- Sprint asymmetry index: |left_stride - right_stride| / avg_stride
- Knee flexion angle: hip-knee-ankle angle at ground contact
- Stride length estimation from ankle keypoints
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# MediaPipe pose landmark indices
LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at point b formed by vectors b→a and b→c (degrees).

    Args:
        a: First point [x, y] or [x, y, z].
        b: Vertex point.
        c: Third point.

    Returns:
        Angle in degrees (0–180).
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _landmark_to_array(lm_obj) -> np.ndarray:
    """Convert a MediaPipe landmark object to numpy [x, y, z]."""
    return np.array([lm_obj.x, lm_obj.y, lm_obj.z])


# ---------------------------------------------------------------------------
# Single-frame pose extraction
# ---------------------------------------------------------------------------

_MODEL_PATH = "/tmp/pose_landmarker_full.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def _ensure_model() -> str:
    """Download the pose landmarker model if not already cached."""
    import urllib.request

    if not Path(_MODEL_PATH).exists():
        logger.info("Downloading MediaPipe pose landmarker model…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _landmarks_to_df(pose_landmarks_list, idx_to_name: dict, frame_idx: int = 0) -> list[dict]:
    """Convert MediaPipe Tasks pose landmark list to a list of row dicts."""
    records = []
    for pose_landmarks in pose_landmarks_list:
        for i, lm_obj in enumerate(pose_landmarks):
            records.append(
                {
                    "frame_idx": frame_idx,
                    "landmark_idx": i,
                    "landmark_name": idx_to_name.get(i, f"lm_{i}"),
                    "x": lm_obj.x,
                    "y": lm_obj.y,
                    "z": lm_obj.z,
                    "visibility": getattr(lm_obj, "visibility", 1.0),
                }
            )
    return records


def extract_keypoints_from_image(
    image_path: str,
    model_complexity: int = 1,
) -> Optional[pd.DataFrame]:
    """Extract 33 MediaPipe pose keypoints from a single image.

    Args:
        image_path: Path to image file (jpg, png, etc.).
        model_complexity: Unused (kept for API compatibility; model is always full).

    Returns:
        DataFrame with columns: landmark_name, landmark_idx, x, y, z, visibility.
        Returns None if no pose detected.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        import cv2
    except ImportError:
        raise ImportError("mediapipe and opencv-python are required: pip install mediapipe opencv-python")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        logger.warning("No pose detected in %s", image_path)
        return None

    idx_to_name = {v: k for k, v in LM.items()}
    records = _landmarks_to_df(result.pose_landmarks, idx_to_name, frame_idx=0)
    return pd.DataFrame(records)


def extract_keypoints_from_video(
    video_path: str,
    sample_every_n_frames: int = 5,
    model_complexity: int = 1,
) -> pd.DataFrame:
    """Extract pose keypoints from every Nth frame of a video.

    Args:
        video_path: Path to video file.
        sample_every_n_frames: Process one frame every N frames.
        model_complexity: Unused (kept for API compatibility).

    Returns:
        DataFrame with all keypoints across sampled frames, plus a 'frame_idx' column.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        import cv2
    except ImportError:
        raise ImportError("mediapipe and opencv-python are required: pip install mediapipe opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    idx_to_name = {v: k for k, v in LM.items()}
    all_records = []
    frame_idx = 0

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model()),
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.PoseLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n_frames == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = detector.detect(mp_image)
                if result.pose_landmarks:
                    all_records.extend(
                        _landmarks_to_df(result.pose_landmarks, idx_to_name, frame_idx)
                    )
            frame_idx += 1

    cap.release()
    df = pd.DataFrame(all_records)
    logger.info(
        "Extracted %d keypoint rows across %d sampled frames from %s",
        len(df),
        frame_idx // sample_every_n_frames,
        video_path,
    )
    return df


# ---------------------------------------------------------------------------
# Biomechanical metrics
# ---------------------------------------------------------------------------

def compute_knee_flexion_angle(
    keypoints_df: pd.DataFrame,
    side: str = "left",
) -> float:
    """Compute knee flexion angle (hip-knee-ankle) from a single-frame keypoint DataFrame.

    Args:
        keypoints_df: DataFrame from extract_keypoints_from_image() or one frame
            of extract_keypoints_from_video().
        side: "left" or "right".

    Returns:
        Knee flexion angle in degrees. Returns np.nan if landmarks not found.
    """
    kp = keypoints_df.set_index("landmark_name")

    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"
    ankle_name = f"{side}_ankle"

    for name in [hip_name, knee_name, ankle_name]:
        if name not in kp.index:
            logger.warning("Landmark %s not found", name)
            return np.nan

    hip = kp.loc[hip_name, ["x", "y", "z"]].values.astype(float)
    knee = kp.loc[knee_name, ["x", "y", "z"]].values.astype(float)
    ankle = kp.loc[ankle_name, ["x", "y", "z"]].values.astype(float)

    return _angle_between(hip, knee, ankle)


def compute_stride_length(
    keypoints_df: pd.DataFrame,
    side: str = "left",
) -> float:
    """Estimate stride length as the distance from heel to foot index (normalized coords).

    Args:
        keypoints_df: Single-frame keypoints DataFrame.
        side: "left" or "right".

    Returns:
        Stride length in normalized image coordinates. Returns np.nan if not found.
    """
    kp = keypoints_df.set_index("landmark_name")
    heel_name = f"{side}_heel"
    foot_name = f"{side}_foot_index"

    if heel_name not in kp.index or foot_name not in kp.index:
        return np.nan

    heel = kp.loc[heel_name, ["x", "y"]].values.astype(float)
    foot = kp.loc[foot_name, ["x", "y"]].values.astype(float)
    return float(np.linalg.norm(foot - heel))


def compute_sprint_asymmetry_index(
    keypoints_sequence: pd.DataFrame,
) -> float:
    """Compute sprint asymmetry index across a sequence of frames.

    Asymmetry = |left_stride - right_stride| / avg_stride * 100 (%)

    Args:
        keypoints_sequence: Multi-frame keypoints DataFrame with 'frame_idx' column.

    Returns:
        Sprint asymmetry index as a percentage. Returns np.nan if insufficient data.
    """
    frame_ids = keypoints_sequence["frame_idx"].unique()
    left_strides = []
    right_strides = []

    for fid in frame_ids:
        frame = keypoints_sequence[keypoints_sequence["frame_idx"] == fid]
        l = compute_stride_length(frame, side="left")
        r = compute_stride_length(frame, side="right")
        if not np.isnan(l):
            left_strides.append(l)
        if not np.isnan(r):
            right_strides.append(r)

    if not left_strides or not right_strides:
        return np.nan

    mean_left = float(np.mean(left_strides))
    mean_right = float(np.mean(right_strides))
    avg = (mean_left + mean_right) / 2
    if avg == 0:
        return np.nan

    asymmetry = abs(mean_left - mean_right) / avg * 100
    logger.info(
        "Sprint asymmetry: %.1f%%  (left=%.4f  right=%.4f)",
        asymmetry,
        mean_left,
        mean_right,
    )
    return float(asymmetry)


def compute_pose_metrics_per_frame(
    keypoints_sequence: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-frame biomechanical metrics from a keypoint sequence.

    For each frame computes:
    - left/right knee flexion angles
    - left/right stride lengths
    - lateral balance (hip midpoint x offset from center)

    Args:
        keypoints_sequence: Multi-frame keypoints DataFrame with 'frame_idx' column.

    Returns:
        DataFrame with one row per frame and metric columns.
    """
    records = []
    for fid in sorted(keypoints_sequence["frame_idx"].unique()):
        frame = keypoints_sequence[keypoints_sequence["frame_idx"] == fid]
        kp = frame.set_index("landmark_name")

        record = {"frame_idx": fid}
        record["left_knee_flexion"] = compute_knee_flexion_angle(frame, "left")
        record["right_knee_flexion"] = compute_knee_flexion_angle(frame, "right")
        record["left_stride_length"] = compute_stride_length(frame, "left")
        record["right_stride_length"] = compute_stride_length(frame, "right")

        # Lateral balance: horizontal distance of hip midpoint from 0.5 (image center)
        if "left_hip" in kp.index and "right_hip" in kp.index:
            hip_mid_x = (kp.loc["left_hip", "x"] + kp.loc["right_hip", "x"]) / 2
            record["lateral_balance_offset"] = float(hip_mid_x - 0.5)
        else:
            record["lateral_balance_offset"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def summarize_pose_metrics(metrics_df: pd.DataFrame) -> dict:
    """Summarize per-frame pose metrics into a player-level report.

    Args:
        metrics_df: Output of compute_pose_metrics_per_frame().

    Returns:
        Dict with mean/std for each metric and the sprint asymmetry index.
    """
    summary = {}
    for col in ["left_knee_flexion", "right_knee_flexion", "left_stride_length", "right_stride_length", "lateral_balance_offset"]:
        if col in metrics_df.columns:
            summary[f"{col}_mean"] = float(metrics_df[col].mean())
            summary[f"{col}_std"] = float(metrics_df[col].std())

    if "left_stride_length" in metrics_df and "right_stride_length" in metrics_df:
        mean_left = metrics_df["left_stride_length"].mean()
        mean_right = metrics_df["right_stride_length"].mean()
        avg = (mean_left + mean_right) / 2
        if avg > 0:
            summary["sprint_asymmetry_index"] = float(abs(mean_left - mean_right) / avg * 100)
        else:
            summary["sprint_asymmetry_index"] = np.nan

    return summary


# ---------------------------------------------------------------------------
# Demo / synthetic test (no video required)
# ---------------------------------------------------------------------------

def _generate_synthetic_keypoints(n_frames: int = 20) -> pd.DataFrame:
    """Generate synthetic keypoint data for testing without a real video.

    Simulates a running motion with slight left/right asymmetry.

    Args:
        n_frames: Number of frames to simulate.

    Returns:
        DataFrame in the same format as extract_keypoints_from_video().
    """
    rng = np.random.default_rng(42)
    records = []
    idx_to_name = {v: k for k, v in LM.items()}

    for fid in range(n_frames):
        phase = fid / n_frames * 2 * np.pi
        for i in range(33):
            name = idx_to_name.get(i, f"lm_{i}")
            # Add a small running oscillation to ankle landmarks
            base_x = 0.5 + rng.normal(0, 0.01)
            base_y = 0.5 + rng.normal(0, 0.01)
            if "left_ankle" in name:
                base_y += 0.05 * np.sin(phase)
            elif "right_ankle" in name:
                base_y += 0.05 * np.sin(phase + np.pi) * 1.1  # 10% asymmetry
            records.append(
                {
                    "frame_idx": fid,
                    "landmark_idx": i,
                    "landmark_name": name,
                    "x": float(base_x),
                    "y": float(base_y),
                    "z": float(rng.normal(0, 0.01)),
                    "visibility": 0.9,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Running pose analysis on synthetic data (no video required)...\n")
    kp_seq = _generate_synthetic_keypoints(n_frames=30)

    metrics_df = compute_pose_metrics_per_frame(kp_seq)
    print("Per-frame metrics (first 5 rows):")
    print(metrics_df.head().to_string(index=False))

    summary = summarize_pose_metrics(metrics_df)
    print("\nPlayer-level summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")

    asym = compute_sprint_asymmetry_index(kp_seq)
    print(f"\nSprint Asymmetry Index: {asym:.2f}%")
