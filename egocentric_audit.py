"""
Task 1 (egocentric) — real-data wrist-camera quality audit.

Decodes every frame of the real `observation.images.cam_left_wrist` stream from
lerobot/aloha_static_coffee (50 episodes x 1100 frames, 480x640, 50fps) with
OpenCV (torchcodec/ffmpeg was not usable in this environment, see README), and
computes real per-frame image-quality metrics:

  - blur / sharpness  : variance of the Laplacian (standard no-reference blur proxy)
  - brightness        : mean grayscale value
  - exposure outliers : fraction of pixels saturated near 0 or 255

These are correlated against the REAL joint-state stream (same episodes) to test
two concrete claims instead of asserting them:
  1. blur tracks the camera's own motion (approximated by left-arm joint velocity,
     since the camera is rigidly mounted to the left wrist)
  2. a simple "dark bottom-of-frame" proxy for gripper self-occlusion actually
     correlates with the real left-gripper state channel (index 6)

Run: python egocentric_audit.py
"""

import json
import time

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DARK_PIXEL_VALUE = 25
BRIGHT_PIXEL_VALUE = 230
LEFT_GRIPPER_STATE_IDX = 6  # left-arm 6 joints (0-5) + left gripper (6); see dataset_audit.py


def load_state():
    path = hf_hub_download("lerobot/aloha_static_coffee",
                            "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].values)
    ep = df["episode_index"].values
    return state, ep


def frame_metrics(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    dark_frac = float((gray < DARK_PIXEL_VALUE).mean())
    bright_frac = float((gray > BRIGHT_PIXEL_VALUE).mean())
    # bottom third of the wrist-cam frame is where a closing gripper/held object
    # most often enters the field of view first
    bottom = gray[gray.shape[0] * 2 // 3:, :]
    bottom_dark_frac = float((bottom < DARK_PIXEL_VALUE + 15).mean())
    return blur, brightness, dark_frac, bright_frac, bottom_dark_frac


def main():
    video_path = hf_hub_download(
        "lerobot/aloha_static_coffee",
        "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        repo_type="dataset",
    )
    state, ep = load_state()
    n_frames = len(ep)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"could not open {video_path}"

    blur = np.zeros(n_frames)
    brightness = np.zeros(n_frames)
    dark_frac = np.zeros(n_frames)
    bright_frac = np.zeros(n_frames)
    bottom_dark_frac = np.zeros(n_frames)

    t0 = time.time()
    i = 0
    while i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        blur[i], brightness[i], dark_frac[i], bright_frac[i], bottom_dark_frac[i] = frame_metrics(frame)
        i += 1
        if i % 10000 == 0:
            print(f"  decoded {i}/{n_frames} frames ({time.time() - t0:.0f}s)")
    cap.release()
    n_decoded = i
    print(f"decoded {n_decoded} real frames in {time.time() - t0:.1f}s")

    blur, brightness = blur[:n_decoded], brightness[:n_decoded]
    dark_frac, bright_frac = dark_frac[:n_decoded], bright_frac[:n_decoded]
    bottom_dark_frac = bottom_dark_frac[:n_decoded]
    state, ep = state[:n_decoded], ep[:n_decoded]

    # blur threshold set data-drivenly at the 15th percentile of this stream's own
    # blur-score distribution, rather than an arbitrary constant
    blur_thresh = float(np.percentile(blur, 15))
    blurred_frac = float((blur < blur_thresh).mean())

    # real wrist angular-velocity proxy: finite-difference norm of the left-arm's
    # 6 joint angles (indices 0-5); camera is rigidly mounted to this arm
    left_arm = state[:, 0:6]
    ang_vel = np.zeros(len(left_arm))
    ang_vel[1:] = np.linalg.norm(np.diff(left_arm, axis=0), axis=1) * 50  # fps=50 -> rad/s
    blur_vs_motion_corr = float(np.corrcoef(blur, -ang_vel)[0, 1])  # expect blur UP as motion DOWN

    gripper = state[:, LEFT_GRIPPER_STATE_IDX]
    occlusion_corr = float(np.corrcoef(bottom_dark_frac, -gripper)[0, 1])

    overexposed_frac = float((bright_frac > 0.05).mean())
    underexposed_frac = float((dark_frac > 0.5).mean())

    # per-episode "review" flag: episode is flagged if >15% of its frames are
    # below the blur threshold (mirrors the review-queue idea from dataset_audit.py)
    episodes = np.unique(ep)
    flagged_episodes = []
    for e in episodes:
        m = ep == e
        frac_blurred = (blur[m] < blur_thresh).mean()
        if frac_blurred > 0.15:
            flagged_episodes.append(int(e))

    results = {
        "video": "lerobot/aloha_static_coffee observation.images.cam_left_wrist",
        "n_frames_decoded": int(n_decoded),
        "n_episodes": int(len(episodes)),
        "blur": {
            "detector": "Laplacian variance, threshold = 15th percentile of this stream",
            "threshold": blur_thresh,
            "frac_frames_below_threshold": blurred_frac,
            "mean": float(blur.mean()), "std": float(blur.std()),
        },
        "blur_vs_wrist_motion_correlation": blur_vs_motion_corr,
        "exposure": {
            "overexposed_frame_frac (>5% bright px)": overexposed_frac,
            "underexposed_frame_frac (>50% dark px)": underexposed_frac,
            "mean_brightness": float(brightness.mean()),
        },
        "occlusion_proxy": {
            "method": "fraction of dark pixels in bottom third of frame vs. real left-gripper state channel",
            "correlation_with_neg_gripper_state": occlusion_corr,
            "note": "heuristic, not verified against pixel-level occlusion labels; "
                    "reported correlation is the actual evidence for/against it, not asserted.",
        },
        "episodes_flagged_for_review (>15% blurred frames)": {
            "episodes": flagged_episodes,
            "count": len(flagged_episodes),
        },
    }
    print(json.dumps(results, indent=2))
    with open("audit_results_egocentric.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote audit_results_egocentric.json")

    # save arrays for plotting
    np.savez("egocentric_frame_metrics.npz", blur=blur, brightness=brightness,
             dark_frac=dark_frac, bright_frac=bright_frac,
             bottom_dark_frac=bottom_dark_frac, ang_vel=ang_vel,
             gripper=gripper, episode_index=ep)


if __name__ == "__main__":
    main()
