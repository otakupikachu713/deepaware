"""
Task 3 — Data curation pipeline, built on the real detectors from Task 1
(dataset_audit.py / egocentric_audit.py), run on the real lerobot/aloha_static_coffee
dataset (the one real dataset in this audit that has both joint-state AND wrist-camera
data, so it's the only one where a joint pipeline curation step is meaningful).

Teleoperation side — two filtering/cleaning steps:
  1. Drop episodes containing a statistical joint-value outlier (|z|>6). One flagged
     episode -> drop the whole episode (real, localized fault found in Task 1: episode 8).
  2. Savitzky-Golay smoothing of observation.state and action, per episode per joint,
     to remove high-frequency encoder/logging jitter without touching the low-frequency
     task trajectory shape.
  (A third step — flagging, not dropping, action/state-desync episodes for manual
   review — is included since Task 1 found that detector isn't reliable enough on this
   dataset to auto-filter; see openarm_dataset_quality_audit.md.)

Egocentric side — extends the pipeline with a real per-frame validity mask from the
wrist camera (blur below the Task 1 data-driven threshold, or overexposed), keyed to
the same (episode_index, frame_index) as the teleop table so alignment is never broken
by physically deleting video frames.

Run: python curation_pipeline.py   (needs audit_results_egocentric.json and
egocentric_frame_metrics.npz from egocentric_audit.py already in this directory)
"""

import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.signal import savgol_filter

OUTLIER_Z = 6.0
SMOOTH_WINDOW = 9   # frames, odd, ~0.18s at 50fps
SMOOTH_POLYORDER = 2
DESYNC_Z = 4.0
DESYNC_MIN_RUN = 10


def load_teleop():
    path = hf_hub_download("lerobot/aloha_static_coffee",
                            "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["observation.state"] = list(np.stack(df["observation.state"].values))
    df["action"] = list(np.stack(df["action"].values))
    return df


def find_outlier_episodes(df):
    state = np.stack(df["observation.state"].values)
    mu, sigma = state.mean(0), state.std(0) + 1e-9
    z = np.abs((state - mu) / sigma)
    outlier_mask = (z > OUTLIER_Z).any(axis=1)
    return sorted(set(df.loc[outlier_mask, "episode_index"].tolist()))


def find_desync_episodes(df):
    state = np.stack(df["observation.state"].values)
    action = np.stack(df["action"].values)
    resid = np.linalg.norm(action - state, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9
    rz = np.abs(resid - med) / (1.4826 * mad)
    ep = df["episode_index"].values
    flagged = set()
    for e in np.unique(ep):
        flags = rz[ep == e] > DESYNC_Z
        run = 0
        for f in flags:
            run = run + 1 if f else 0
            if run >= DESYNC_MIN_RUN:
                flagged.add(int(e))
                break
    return sorted(flagged)


def smooth_episode(sub_df):
    n = len(sub_df)
    win = min(SMOOTH_WINDOW, n - (1 - n % 2))  # keep window odd and <= episode length
    if win < SMOOTH_POLYORDER + 2:
        return sub_df  # too short to smooth safely, leave as-is
    if win % 2 == 0:
        win -= 1
    for col in ("observation.state", "action"):
        arr = np.stack(sub_df[col].values)
        smoothed = savgol_filter(arr, window_length=win, polyorder=SMOOTH_POLYORDER, axis=0)
        sub_df = sub_df.copy()
        sub_df[col] = list(smoothed)
    return sub_df


def build_egocentric_mask():
    with open("audit_results_egocentric.json") as f:
        egoc = json.load(f)
    blur_thresh = egoc["blur"]["threshold"]
    d = np.load("egocentric_frame_metrics.npz")
    blurred = d["blur"] < blur_thresh
    overexposed = d["bright_frac"] > 0.05
    valid = ~(blurred | overexposed)
    mask_df = pd.DataFrame({
        "episode_index": d["episode_index"],
        "video_valid": valid,
        "blurred": blurred,
        "overexposed": overexposed,
    })
    mask_df["frame_index"] = mask_df.groupby("episode_index").cumcount()
    return mask_df


def main():
    df = load_teleop()
    n_total_frames = len(df)
    n_total_episodes = df["episode_index"].nunique()

    outlier_eps = find_outlier_episodes(df)
    desync_eps = find_desync_episodes(df)

    kept = df[~df["episode_index"].isin(outlier_eps)].copy()
    print(f"step 1 (drop outlier episodes {outlier_eps}): "
          f"{n_total_episodes} -> {kept['episode_index'].nunique()} episodes, "
          f"{n_total_frames} -> {len(kept)} frames")

    smoothed_parts = []
    for e, g in kept.groupby("episode_index", sort=True):
        smoothed_parts.append(smooth_episode(g))
    kept = pd.concat(smoothed_parts, ignore_index=True)
    print(f"step 2: smoothed observation.state/action for "
          f"{kept['episode_index'].nunique()} retained episodes "
          f"(Savitzky-Golay, window={SMOOTH_WINDOW}, polyorder={SMOOTH_POLYORDER})")

    print(f"desync review queue (flagged, NOT auto-dropped, per Task 1 finding "
          f"that this detector isn't reliable enough to trust blindly): {desync_eps}")

    # --- egocentric extension: real per-frame validity mask, joined on (episode, frame) ---
    mask_df = build_egocentric_mask()
    kept["frame_index_in_episode"] = kept.groupby("episode_index").cumcount()
    joined = kept.merge(
        mask_df.rename(columns={"frame_index": "frame_index_in_episode"}),
        on=["episode_index", "frame_index_in_episode"], how="left",
    )
    n_video_valid = int(joined["video_valid"].sum())
    n_video_total = len(joined)

    curated_state = pd.DataFrame({
        "episode_index": joined["episode_index"],
        "frame_index": joined["frame_index_in_episode"],
        "timestamp": joined["timestamp"],
        "video_valid": joined["video_valid"],
    })
    curated_state.to_parquet("curated_teleop_frame_index.parquet")
    np.savez("curated_teleop_arrays.npz",
             episode_index=joined["episode_index"].values,
             frame_index=joined["frame_index_in_episode"].values,
             state=np.stack(joined["observation.state"].values),
             action=np.stack(joined["action"].values),
             video_valid=joined["video_valid"].values)

    report = {
        "source_dataset": "lerobot/aloha_static_coffee",
        "input": {"episodes": n_total_episodes, "frames": n_total_frames},
        "step1_drop_outlier_episodes": {"dropped_episodes": outlier_eps,
                                          "justification": "episode contains a joint-state "
                                          "value |z|>6 vs. that joint's own global distribution "
                                          "(real finding from Task 1, episode 8, 56 frames) - "
                                          "physically implausible values are almost never "
                                          "recoverable per-frame, drop the whole recording session"},
        "step2_smoothing": {"method": "Savitzky-Golay", "window": SMOOTH_WINDOW,
                              "polyorder": SMOOTH_POLYORDER,
                              "justification": "reduces high-frequency encoder/logging jitter "
                              "in state/action without distorting the low-frequency task "
                              "trajectory shape (polyorder=2 preserves smooth accel/decel)"},
        "desync_review_queue_not_auto_dropped": desync_eps,
        "egocentric_extension": {
            "method": "per-frame validity mask (blurred OR overexposed) from egocentric_audit.py, "
                       "joined on (episode_index, frame_index) rather than deleting video frames, "
                       "so alignment with the (filtered, smoothed) teleop table is never broken",
            "frames_after_teleop_filtering": n_video_total,
            "frames_also_video_valid": n_video_valid,
            "video_valid_fraction": n_video_valid / n_video_total,
        },
        "output": {
            "episodes": int(kept["episode_index"].nunique()),
            "frames": len(kept),
            "frames_usable_by_both_modalities": n_video_valid,
        },
    }
    with open("curation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("wrote curated_teleop_arrays.npz, curated_teleop_frame_index.parquet, curation_report.json")


if __name__ == "__main__":
    main()
