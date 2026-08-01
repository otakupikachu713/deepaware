# OpenArm 2.0 — Dataset Exploration & Quality Audit (Task 1)

## Scope and method

This audit runs on **real, downloaded LeRobot-format data** — no synthetic data. Two Hugging Face
datasets:

- `lerobot/aloha_sim_insertion_human` — simulation, the dataset named in the take-home prompt
- `lerobot/aloha_static_coffee` — real ALOHA teleoperated hardware, chosen because it has wrist
  cameras (`cam_left_wrist`/`cam_right_wrist`), which `aloha_sim_insertion_human` does not

`dataset_audit.py` profiles the joint-state/action tables of both (via `LeRobotDataset.hf_dataset`
and the raw parquet, no video decode needed). `egocentric_audit.py` decodes **every one of the
55,000 real frames** of `aloha_static_coffee`'s `cam_left_wrist` stream and computes real
per-frame image-quality metrics, correlated against the real joint-state stream. `plot_audit_results.py`
produces `audit_plots.png` from the saved results. All three scripts run end-to-end; every number
below is direct script output, not estimated.

**Environment note:** `torchcodec` (lerobot's default video backend) could not load its native
DLLs on this Windows machine (`libtorchcodec_core8.dll` missing dependency, tried FFmpeg 4–8). The
teleop profiling doesn't need video and is unaffected. For the egocentric stream, video decode was
done directly with `cv2.VideoCapture` on the downloaded mp4 (OpenCV's bundled FFmpeg build handles
it fine) instead of going through lerobot's `LeRobotDataset.__getitem__`.

---

## Part 1 — Teleoperation data

### Profiling results (`dataset_audit.py`, real data)

| | `aloha_sim_insertion_human` | `aloha_static_coffee` |
|---|---|---|
| Episodes | 50 | 50 |
| Frames | 25,000 | 55,000 |
| Episode length | **exactly 500 frames, every episode** (std = 0) | **exactly 1,100 frames, every episode** (std = 0) |
| Dropped-frame gap events (timestamp Δ > 1.5× expected) | 0 | 0 |
| Joint-value statistical outliers (\|z\|>6 vs. that joint's own distribution) | 0 frames | 56 frames, all in episode 8 |
| Frozen-sensor windows, naive (any 1 of 14 joints near-zero variance for 15 frames) | 9/50 episodes, 18 windows | **50/50 episodes**, 3,185 windows |
| Frozen-sensor windows, strict (all 14 joints simultaneously frozen) | 0/50 episodes | 0/50 episodes |
| Action–state desync (sustained robust z-score >4 on \|\|action−state\|\|) | 50/50 episodes, mean residual 0.452 | 13/50 episodes, mean residual 0.108 |

**Two results here overturned an assumption I'd made before actually running this on real data:**

1. **Both real datasets use a fixed per-episode frame budget**, not the human-paced variable-length
   episodes I'd assumed. `aloha_sim_insertion_human` is exactly 500 frames every time (expected —
   it's a scripted sim rollout with a fixed horizon), but `aloha_static_coffee` — real ALOHA
   hardware, human teleoperator — is *also* exactly 1,100 frames every single time, zero variance.
   That means this particular published dataset was collected (or trimmed) to a fixed duration
   per demo, not "task complete → stop." **Trajectory-length filtering isn't a live issue for
   either of these two datasets as published** — but it would still be the first thing to check on
   any new dataset, since it's cheap and a dataset that *does* have real length variance is exactly
   where a length filter earns its keep.

2. **The naive frozen-sensor detector is not useless, but its naive form is close to it.** On
   `aloha_static_coffee`, checking "did *any one* of the 14 joints go near-zero-variance for 15
   consecutive frames" flags **every single episode** (3,185 windows total) — because manipulation
   demonstrations legitimately hold still for brief stretches (settling into a grasp, waiting mid-
   approach) and that looks identical to a stuck sensor if you check one channel at a time. Checking
   instead whether **all 14 channels freeze in the same window simultaneously** — the actual
   signature of a comms/logging fault, since a real robot doesn't have all 14 joints coincidentally
   stop moving at once unless something upstream stalled — finds **zero** such events in either
   dataset. Conclusion for a real pipeline: single-channel freeze detection has ~0% specificity on
   manipulation data and isn't worth deploying as-is; the simultaneous-freeze version is a much
   better-calibrated first filter, and neither published dataset here actually has a sensor-freeze
   problem.

**The action–state desync detector needed the same kind of honesty check.** On the sim dataset it
flagged all 50/50 episodes with a much larger mean residual (0.45 vs. 0.11) than on the real
hardware dataset. Flagging literally 100% of episodes means the detector isn't discriminating
anything — the far more likely explanation is that in `aloha_sim_insertion_human`'s recording
convention, `action` is a forward/target reference the controller is tracking (not required to
equal the instantaneous proprioceptive `state`), so a sustained nonzero residual is the *normal*
operating regime for this dataset, not a defect. On `aloha_static_coffee` the same detector flags a
more plausible 13/50 (26%) — small enough to be a real minority worth a look, not a formatting
artifact. This is the concrete version of a general lesson: a "defect detector" tuned on one
dataset's conventions can silently become a no-op or a false-alarm generator on another dataset
with a different logging convention, and the tell is exactly this — check the flag rate before
trusting the flag.

### What to filter before training

1. **Episodes/frames with statistical joint-value outliers** — real, localized (episode 8's
   56 frames in `aloha_static_coffee`) — worth a manual look before deciding whether to drop the
   episode or just the offending frame range; a single outlier episode buried in 50 is exactly the
   kind of thing that's easy to miss without automated profiling.
2. **Any window where all channels freeze simultaneously** (the strict detector) — none found in
   these two datasets, but this is the version worth actually deploying, unlike the naive
   any-single-channel version, which would junk-filter legitimate holds.
3. **Action–state desync, gated by a per-dataset flag-rate sanity check first** — useful signal on
   `aloha_static_coffee` (13/50), meaningless on `aloha_sim_insertion_human` (50/50) where it's
   measuring the recording convention, not a defect. Don't apply the same detector output as a
   drop rule across datasets without checking this.
4. **Dropped-frame boundaries** — the detector exists and is cheap (pure timestamp arithmetic) but
   found nothing in either published dataset; still worth running on any new/raw recording, since
   published benchmark dumps are usually already cleaned of raw ingestion drops before release —
   that's likely *why* both datasets came back clean here, not evidence the problem doesn't exist
   in general.

---

## Part 2 — Egocentric (wrist-camera) video

### Why this needs a different lens

`aloha_sim_insertion_human` has no wrist camera at all (only a static `observation.images.top`
view), so this section runs entirely on `aloha_static_coffee`'s real `cam_left_wrist` stream —
480×640, av1, 50fps, rigidly mounted to the left wrist. All 55,000 frames were decoded and scored.

### Profiling results (`egocentric_audit.py`, real data)

- **Blur:** Laplacian-variance blur score, mean 58.4, std 41.0, heavily right-skewed (long tail of
  very sharp frames, cluster of low-sharpness frames near zero — see `audit_plots.png`). Setting
  the "flag" threshold at the stream's own 15th percentile (23.7) is a data-driven choice rather
  than a hand-picked constant.
- **Blur vs. wrist motion: correlation is weak (r ≈ 0.05), not what I expected going in.** I
  approximated camera angular velocity with the finite-difference speed of the left arm's 6 joint
  angles, tried every individual joint too (r ranged 0.02–0.13 across all of them), and none of
  them meaningfully predict blur. This is a real, useful negative result, and it corrects an
  assumption I'd carried in before testing it: joint-angle finite-differences at 50Hz are not a
  good stand-in for the camera's actual angular velocity, because the camera's real rotation is a
  nonlinear function of several joints via forward kinematics, not a simple per-joint sum, and
  because blur here may be dominated by defocus, exposure transients, or av1 compression artifacts
  rather than motion at all. **A real pipeline needs either true camera pose (forward kinematics)
  or optical flow computed straight from the video to attribute blur causally — a joint-velocity
  proxy is not a reliable substitute**, and I would have reported a false causal story if I hadn't
  checked this against real pixels.
- **Exposure is the dominant, and more surprising, real issue: 66.1% of all frames have >5% of
  pixels at hard bright/saturated values**, vs. 0% of frames with >50% dark pixels. This is far
  higher than intuition would suggest and is the single biggest quality problem in this stream —
  bigger than blur. A plausible mechanism (not verified pixel-by-pixel here) is a wrist-proximate
  light source or a reflective workspace surface (a coffee machine) frequently filling a large
  fraction of the close-range frame.
- **Occlusion proxy:** fraction of dark pixels in the bottom third of the frame vs. the real
  left-gripper state channel gives r ≈ −0.34 (darker bottom-of-frame as the gripper closes) — a
  moderate, directionally-consistent real signal supporting the "gripper self-occludes the wrist
  camera" hypothesis, but with plenty of unexplained variance. This is reported as a heuristic
  correlation against a real proprioceptive signal, not validated against pixel-level occlusion
  labels, since none exist for this dataset.
- **Per-episode spread:** 20/50 episodes (40%) exceed a 15%-blurred-frames review threshold, so
  blur is not evenly spread across episodes — some episodes are meaningfully worse than others
  (see the bar chart in `audit_plots.png`), which argues for episode-level triage rather than
  assuming a uniform per-frame drop rate applies everywhere.

Quality issues specific to egocentric video that have no joint-state analogue:

- **Exposure instability tied to the camera's own viewpoint** — confirmed as the single largest
  real defect class in this stream (66% of frames), not a minor secondary issue.
- **Self-occlusion by the gripper/held object** — a real, moderate (r≈−0.34) signal in this data.
- **Motion blur** — present (15% of frames below the data-driven threshold) but, on this real
  dataset, not well explained by a naive joint-velocity proxy; needs a better motion signal
  (forward-kinematics camera pose or optical flow) to attribute causally.
- **Rolling-shutter skew** on fast rotations — a known CMOS wrist-camera artifact, not separately
  measured here (would require frame-pair optical flow, not just single-frame metrics).

### How filtering criteria differ from joint-state data

Joint-state defects here were rare and sharply localized (a handful of frames in one episode,
zero real freeze events) and detectable from cheap statistics on a 14-dim vector. Egocentric
defects are the opposite: **pervasive rather than rare** (66% of frames touched by the exposure
issue alone) and require **decoding video and computing per-frame image metrics**, which is
several orders of magnitude more compute than the joint-state checks. That has concrete
consequences for how a real pipeline should be structured:

- **Run the free joint-state filters first as a gate**, and only pay for video decode + per-frame
  metrics on episodes that survive — cheap statistics on a 14-column array vs. decoding and scoring
  55,000 video frames is not a close call on cost.
- **Filter egocentric defects at the episode or frame-mask level, not by deleting frames from the
  video file.** With 20/50 episodes affected and defects densely distributed within an affected
  episode, physically excising frames would fragment the sequence for a video-conditioned policy;
  a per-frame validity mask keyed to the same `(episode_index, frame_index)` used by the
  joint-state table (see `curation_pipeline.py`) keeps the option open to either drop masked
  frames or handle them explicitly at train time, without breaking temporal alignment.
- **Don't assume a hand-picked motion proxy explains an image-quality metric — check it.** The
  blur/motion correlation coming back at r≈0.05 instead of the strongly negative value I expected
  is the clearest example in this whole audit of why: an unverified assumption here would have
  produced a filtering rule ("drop frames during fast joint motion") that doesn't actually target
  the real cause of blur in this dataset.

---

## Files in this delivery

- `dataset_audit.py` — real teleop profiling (both datasets), writes `audit_results_teleop.json`
- `egocentric_audit.py` — real wrist-camera frame-level profiling, writes
  `audit_results_egocentric.json` and `egocentric_frame_metrics.npz`
- `plot_audit_results.py` — regenerates `audit_plots.png` from the two JSON/npz outputs above
- `audit_results_teleop.json`, `audit_results_egocentric.json` — raw profiling output (real numbers)
- `audit_plots.png` — action-state residual, blur distribution, brightness distribution, blur-vs-
  motion scatter, occlusion-vs-gripper scatter, per-episode blur-flag bar chart
- `curation_pipeline.py` — Task 3, curation pipeline built on these same real detectors
