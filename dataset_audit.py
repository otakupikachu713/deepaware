"""
Task 1 (teleoperation) — real-data quality audit.

Loads two real LeRobot-format datasets from Hugging Face and profiles them:
  - lerobot/aloha_sim_insertion_human  (simulation, named in the take-home prompt)
  - lerobot/aloha_static_coffee        (real ALOHA teleop, has wrist cameras -> used
                                         again in egocentric_audit.py for Part 2)

Only tabular columns (state/action/timestamp/episode_index) are read here, via
`LeRobotDataset.hf_dataset` / the raw parquet file, so no video decoding is needed
for the teleoperation profiling.

Run: python dataset_audit.py
"""

import json
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FREEZE_WINDOW = 15          # frames, ~0.3s at 50fps
FREEZE_STD_TOL = 1e-6       # per-joint std below this counts as "not moving"
DESYNC_MIN_RUN = 10         # frames a residual spike must sustain to count
DESYNC_Z = 4.0              # robust z-score threshold on action-state residual


def load_aloha_sim():
    ds = LeRobotDataset("lerobot/aloha_sim_insertion_human")
    hf = ds.hf_dataset
    hf.set_format("numpy")
    state = np.asarray(hf["observation.state"])
    action = np.asarray(hf["action"])
    ts = np.asarray(hf["timestamp"]).astype(np.float64)
    ep = np.asarray(hf["episode_index"])
    return dict(name="aloha_sim_insertion_human", state=state, action=action,
                timestamp=ts, episode_index=ep, fps=ds.fps)


def load_aloha_static_coffee():
    path = hf_hub_download("lerobot/aloha_static_coffee",
                            "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].values)
    action = np.stack(df["action"].values)
    ts = df["timestamp"].values.astype(np.float64)
    ep = df["episode_index"].values
    return dict(name="aloha_static_coffee", state=state, action=action,
                timestamp=ts, episode_index=ep, fps=50)


def profile(d):
    state, action, ts, ep, fps = d["state"], d["action"], d["timestamp"], d["episode_index"], d["fps"]
    episodes = np.unique(ep)
    lengths = np.array([np.sum(ep == e) for e in episodes])

    # --- dropped frames: within-episode timestamp gaps far from the expected 1/fps step
    expected_dt = 1.0 / fps
    drop_events = []
    for e in episodes:
        mask = ep == e
        t = ts[mask]
        dt = np.diff(t)
        gap_idx = np.where(dt > 1.5 * expected_dt)[0]
        for gi in gap_idx:
            drop_events.append({"episode": int(e), "at_frame": int(gi), "gap_s": float(dt[gi])})

    # --- statistical joint-value outliers: per-joint global mean/std, flag |z| > 6
    mu, sigma = state.mean(0), state.std(0) + 1e-9
    z = np.abs((state - mu) / sigma)
    outlier_frame_mask = (z > 6).any(axis=1)
    outlier_episodes = sorted(set(int(e) for e in ep[outlier_frame_mask]))

    # --- frozen sensor, two variants ---
    # (a) naive: ANY single joint with near-zero rolling std over a window -> flag
    # (b) strict: ALL joints simultaneously frozen in the same window -> real fault signature
    naive_frozen_eps, strict_frozen_eps = set(), set()
    naive_frozen_windows = 0
    for e in episodes:
        s = state[ep == e]
        n = len(s)
        for i in range(0, n - FREEZE_WINDOW, FREEZE_WINDOW):
            w = s[i:i + FREEZE_WINDOW]
            stds = w.std(0)
            if (stds < FREEZE_STD_TOL).any():
                naive_frozen_eps.add(int(e))
                naive_frozen_windows += 1
            if (stds < FREEZE_STD_TOL).all():
                strict_frozen_eps.add(int(e))

    # --- action/state desync: robust z-score on ||action - state|| residual, sustained run
    resid = np.linalg.norm(action - state, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9
    rz = np.abs(resid - med) / (1.4826 * mad)
    desync_eps = set()
    for e in episodes:
        flags = rz[ep == e] > DESYNC_Z
        run = 0
        for f in flags:
            run = run + 1 if f else 0
            if run >= DESYNC_MIN_RUN:
                desync_eps.add(int(e))
                break

    return {
        "dataset": d["name"],
        "num_episodes": int(len(episodes)),
        "num_frames": int(len(ep)),
        "fps": fps,
        "trajectory_length_frames": {
            "mean": float(lengths.mean()), "std": float(lengths.std()),
            "min": int(lengths.min()), "max": int(lengths.max()),
        },
        "dropped_frames": {
            "num_gap_events": len(drop_events),
            "episodes_affected": sorted(set(x["episode"] for x in drop_events)),
            "examples": drop_events[:5],
        },
        "joint_value_outliers": {
            "detector": "|z| > 6 vs. that joint's own global mean/std (14-dim state)",
            "num_outlier_frames": int(outlier_frame_mask.sum()),
            "episodes_affected": outlier_episodes,
        },
        "frozen_sensor": {
            "window_frames": FREEZE_WINDOW,
            "naive_any_joint_frozen": {
                "episodes_flagged": sorted(naive_frozen_eps),
                "num_episodes_flagged": len(naive_frozen_eps),
                "num_windows_flagged": naive_frozen_windows,
                "note": "flags if ANY single joint has near-zero variance in a window; "
                        "on real data this is dominated by legitimate holds/pauses, not faults.",
            },
            "strict_all_joints_simultaneously_frozen": {
                "episodes_flagged": sorted(strict_frozen_eps),
                "num_episodes_flagged": len(strict_frozen_eps),
                "note": "flags only if ALL 14 joints freeze in the same window at once - "
                        "the actual signature of a comms/logging dropout.",
            },
        },
        "action_state_desync": {
            "detector": f"robust z-score(|action-state| residual) > {DESYNC_Z}, sustained >= {DESYNC_MIN_RUN} frames",
            "episodes_flagged": sorted(desync_eps),
            "num_episodes_flagged": len(desync_eps),
            "residual_stats": {"mean": float(resid.mean()), "p99": float(np.percentile(resid, 99)),
                                "max": float(resid.max())},
        },
    }


def main():
    results = {}
    for loader in (load_aloha_sim, load_aloha_static_coffee):
        d = loader()
        print(f"loaded {d['name']}: {len(np.unique(d['episode_index']))} episodes, "
              f"{len(d['episode_index'])} frames")
        results[d["name"]] = profile(d)

    with open("audit_results_teleop.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote audit_results_teleop.json")


if __name__ == "__main__":
    main()
