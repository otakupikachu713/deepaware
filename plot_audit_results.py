"""
Generates audit_plots.png from the real outputs of dataset_audit.py and
egocentric_audit.py (audit_results_*.json, egocentric_frame_metrics.npz).

Run after both audit scripts: python plot_audit_results.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    d = np.load("egocentric_frame_metrics.npz")
    blur, bright_frac = d["blur"], d["bright_frac"]
    bottom_dark, ang_vel = d["bottom_dark_frac"], d["ang_vel"]
    gripper, ep = d["gripper"], d["episode_index"]

    with open("audit_results_egocentric.json") as f:
        egoc = json.load(f)

    path = hf_hub_download("lerobot/aloha_static_coffee",
                            "data/chunk-000/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].values)
    action = np.stack(df["action"].values)
    resid_coffee = np.linalg.norm(action - state, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(resid_coffee, bins=80, color="steelblue")
    axes[0, 0].set_title("Teleop: action-state residual\n(aloha_static_coffee, real, N=55000)")
    axes[0, 0].set_xlabel("||action - state||")
    axes[0, 0].set_ylabel("frame count")

    thresh = egoc["blur"]["threshold"]
    axes[0, 1].hist(blur, bins=100, color="indianred")
    axes[0, 1].axvline(thresh, color="k", linestyle="--", label=f"15th pct = {thresh:.1f}")
    axes[0, 1].set_title("Egocentric: Laplacian-variance blur score\n(cam_left_wrist, real, N=55000)")
    axes[0, 1].set_xlabel("blur score (higher=sharper)")
    axes[0, 1].legend()

    axes[0, 2].hist(d["brightness"], bins=80, color="goldenrod")
    axes[0, 2].set_title("Egocentric: mean frame brightness")
    axes[0, 2].set_xlabel("mean grayscale value (0-255)")

    idx = np.random.RandomState(0).choice(len(blur), 4000, replace=False)
    axes[1, 0].scatter(ang_vel[idx], blur[idx], s=2, alpha=0.3, color="purple")
    r = egoc["blur_vs_wrist_motion_correlation"]
    axes[1, 0].set_title(f"Blur vs left-arm joint speed proxy\n(r={r:.2f} — weak, see write-up)")
    axes[1, 0].set_xlabel("|d(joint)/dt| proxy (rad/s)")
    axes[1, 0].set_ylabel("blur score")

    axes[1, 1].scatter(gripper[idx], bottom_dark[idx], s=2, alpha=0.3, color="darkgreen")
    r2 = egoc["occlusion_proxy"]["correlation_with_neg_gripper_state"]
    axes[1, 1].set_title(f"Bottom-of-frame darkness vs gripper state\n(r={r2:.2f} vs -gripper)")
    axes[1, 1].set_xlabel("left gripper state value")
    axes[1, 1].set_ylabel("frac dark px, bottom third")

    episodes = np.unique(ep)
    fracs = [(blur[ep == e] < thresh).mean() for e in episodes]
    colors = ["crimson" if f > 0.15 else "steelblue" for f in fracs]
    axes[1, 2].bar(episodes, fracs, color=colors)
    axes[1, 2].axhline(0.15, color="k", linestyle="--", label="review threshold")
    axes[1, 2].set_title("Per-episode blurred-frame fraction\n(red = flagged for review)")
    axes[1, 2].set_xlabel("episode index")
    axes[1, 2].set_ylabel("frac frames below blur threshold")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("audit_plots.png", dpi=110)
    print("saved audit_plots.png")


if __name__ == "__main__":
    main()
