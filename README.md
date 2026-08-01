# OpenArm 2.0 — Egocentric Data & Teleoperation Pipeline

## Task 1 — Dataset Exploration & Quality Audit

Full write-up with every number and the reasoning behind it:
[openarm_dataset_quality_audit.md](openarm_dataset_quality_audit.md). Code: `dataset_audit.py`
(teleop), `egocentric_audit.py` (wrist camera), `plot_audit_results.py` (→ `audit_plots.png`). Real
data from `lerobot/aloha_sim_insertion_human` (the dataset named in the prompt) and
`lerobot/aloha_static_coffee` (real ALOHA hardware, has the wrist cameras the sim dataset lacks).
Everything below is real script output against those two datasets — nothing estimated.

### Teleoperation

| | `aloha_sim_insertion_human` | `aloha_static_coffee` |
|---|---|---|
| Episodes / frames | 50 / 25,000 | 50 / 55,000 |
| Episode length | fixed, 500 frames every time | fixed, 1,100 frames every time |
| Dropped-frame gaps | 0 | 0 |
| Joint-value outliers (\|z\|>6) | 0 | 56 frames, all episode 8 |
| Frozen-sensor, naive (any 1/14 joints) | 9/50 episodes | 50/50 episodes |
| Frozen-sensor, strict (all 14 simultaneously) | 0/50 episodes | 0/50 episodes |
| Action–state desync (sustained z>4) | 50/50 episodes | 13/50 episodes |

Two things here overturned an assumption I'd have otherwise stated as fact: **episode length is
fixed, not human-paced-variable, in both datasets** (contrary to what I'd assumed going in), and
the **naive frozen-sensor detector is close to useless on real data** — checking one joint at a
time flags every single episode because manipulation demos legitimately hold still sometimes;
requiring all 14 channels to freeze *simultaneously* is what actually finds the real fault
signature, and finds zero in either dataset. The desync detector flagging 100% of the sim dataset
vs. 13/50 (26%) of the real one is the same lesson again — a 100% flag rate means the detector is
measuring a recording convention, not a defect.

**What I'd filter before training:** episodes with statistical joint-value outliers (found: 1),
windows where all channels freeze simultaneously (found: 0, but this is the version worth
deploying), and action–state desync — gated behind a sanity check on the flag rate per dataset,
since the same detector is meaningless on one dataset and useful on another.

### Egocentric

All 55,000 real frames of `cam_left_wrist` decoded and scored (OpenCV, not torchcodec — see
below):

- **Blur:** mean Laplacian-variance score 58.4, 15% of frames below a data-driven (15th-percentile)
  threshold.
- **Blur vs. wrist motion:** r ≈ 0.05 — essentially uncorrelated. I'd assumed motion blur would
  track joint velocity strongly; a finite-difference joint-velocity proxy turned out not to explain
  it at all, tested against every individual joint too (r ranged 0.02–0.13). Real negative result,
  not swept under the rug.
- **Exposure — the actual dominant defect:** 66.1% of frames have >5% saturated/bright pixels,
  0% have >50% dark pixels. Bigger and more surprising than the blur problem.
- **Occlusion proxy** (bottom-of-frame darkness vs. real left-gripper state): r ≈ −0.34, a moderate
  real signal supporting the "gripper self-occludes the wrist camera" hypothesis.
- 20/50 episodes (40%) exceed a 15%-blurred-frames review threshold — blur is unevenly distributed
  across episodes, not a uniform per-frame nuisance.

**Environment note:** `torchcodec` (lerobot's default video backend) couldn't load its native DLLs
on this Windows machine for any FFmpeg version it tried. Video was decoded directly with
`cv2.VideoCapture` instead, which bundles a working FFmpeg build — teleop profiling was unaffected
since it doesn't touch video at all.

---

## Task 2 — Data Labeling Design

### Teleoperation

**Schema:** mainly segment-level, with episode-level labels auto-generated from the segment-level
analysis rather than labeled separately. Each move gets a label — reach, lift, transport, place,
release — combined with whether it succeeded, so an example segment-level label looks like
`reach-success`. At the episode level, this rolls up into an overall success/failure label plus a
failure reason, where the failure reason itself is derived from the segment-level breakdown instead
of being hand-labeled again from scratch.

**Tool:** if we can modify CVAT's frontend to support multi-camera labeling, use CVAT — it's more
efficient for frame interpolation and object tracking. If that engineering cost isn't available,
use Label Studio instead. (CVAT doesn't ship synchronized multi-camera viewing today — see the
correction below — so "modify the frontend" is a real prerequisite, not a nice-to-have, for ALOHA's
4-camera episodes.)

**Inter-annotator agreement:** use temporal IoU to check whether two annotators actually agree on a
given segment. If the IoU is over 0.8 *and* the label content matches, count it as a valid/agreed
data point.

*Correction to an earlier draft of this section:* I originally stated that CVAT "handles
multi-camera video natively." That's wrong — **CVAT does not natively support synchronized
multi-camera viewing**, this is an open feature request
([cvat-ai/cvat#4915](https://github.com/cvat-ai/cvat/issues/4915)), not a shipped capability. ALOHA
episodes here have up to 4 synchronized camera streams (`cam_high`, `cam_low`, `cam_left_wrist`,
`cam_right_wrist`) that the annotator needs together to judge grasp success (e.g. confirming
contact from a wrist camera while judging placement from the overhead view). Absent the frontend
work above, the practical fallback is to **tile the camera streams into a single composite video
before import** — that tiling step should be budgeted as part of the labeling pipeline, not
treated as free.

### Egocentric

**Object interactions:** label the contact state as `no contact / approaching / in contact /
releasing`.

**Hand-eye coordination phases:** label as `alignment / approaching / grasping / transport`.

**Failure:** pick from a fixed set of common failure-triggering events — `misalignment / miss
grasp / visual disturbance`.

**Gaze proxy regions:** label the salient keypoint and an attention heatmap, to see whether the arm
is actually focused on the right object.

**Temporal alignment:** document the offset in the time boundary between modalities, and use
episode index + frame index as the frame key.

**Addition — using the real force/effort signal to cut labeling cost:** the `no contact →
approaching → in contact → releasing` states above don't have to be hand-labeled from video alone.
`aloha_static_coffee` has a real `observation.effort` channel (verified against the actual
downloaded data, not assumed), and it's close to uncorrelated with gripper position (r≈0.05 between
left-gripper effort and left-gripper state) — meaning it's carrying genuinely independent
information, not a redundant copy of position. A spike in gripper effort is a strong candidate
boundary for `in contact`, so the workflow can be: auto-generate candidate contact-state boundaries
from the effort signal, then have the annotator confirm/adjust against video, instead of scrubbing
every episode from scratch.

---

## Task 3 — Data Curation Pipeline

Code: `curation_pipeline.py`, run on the real `aloha_static_coffee` data (the one dataset here with
both joint-state and wrist-camera streams). Uses the exact detectors validated in Task 1, not a
separate ad hoc set. Output: `curated_teleop_arrays.npz`, `curated_teleop_frame_index.parquet`,
`curation_report.json`.

### Teleoperation — two filtering/cleaning steps

1. **Drop episodes with a statistical joint-value outlier** (Task 1's real finding: episode 8,
   56 frames, |z|>6) — 50 → 49 episodes, 55,000 → 53,900 frames. Justification: a physically
   implausible joint value is almost never recoverable per-frame; dropping the whole recording
   session is cheap (1/50 = 2% of the data) against the risk of training on a corrupted episode.
2. **Savitzky-Golay smoothing** (window=9, polyorder=2) on `observation.state`/`action` for every
   retained episode. Justification: reduces high-frequency encoder/logging jitter without
   distorting the low-frequency task trajectory shape (polyorder=2 preserves smooth accel/decel).

A third step is **flag, don't drop**: the 13 action–state-desync episodes from Task 1 go into a
manual review queue rather than being auto-filtered, since Task 1 showed that detector isn't
reliable enough on this dataset to trust blindly.

### Egocentric extension

A real per-frame validity mask (blurred OR overexposed, using Task 1's data-driven thresholds) is
joined onto the teleop table on `(episode_index, frame_index)` rather than physically deleting
frames from the video file — this keeps the two modalities aligned without fragmenting the video
sequence or duplicating decoded frames on disk.

### Result

Of the 53,900 frames that survive the teleop filtering step, only **15,501 (28.8%) are also
video-valid** — a real, fairly stark number that reflects Task 1's finding that the egocentric
stream (dominated by the 66% overexposure issue) is a much bigger source of unusable data than the
joint-state stream ever was.

---

## Task 4 — Policy Evaluation Design

### Teleoperation

**Metrics:** task success rate, per-segment success rate (using the Task 2 segment schema), and
completion time for successful tasks. This combination is what lets us diagnose *where* the real
problem is — if the overall failure rate is high, the per-segment breakdown tells us exactly what's
going wrong instead of just that something is.

**Rollout count:** 50 — enough that the success rate isn't an outlier result. Statistically, more
rollouts make the result more valid.

**Success criteria:** a strict rule set, not a human decision, based on:
- **Pose Tolerance** — distance error < 1cm, angle error < 5°
- **Gripper Action + Physical Stability** — full release, held at the place area for 5 seconds
- **Step Limit** — the robot should reach the goal efficiently rather than oscillating back and
  forth (task-dependent)
- **Grasp Force Confirmation** *(addition)* — sustained gripper effort above a threshold during
  the transport phase, confirming the gripper is actually load-bearing an object rather than just
  showing a "closed" position with nothing, or a slipped object, inside. This is a cheap,
  independent check on top of the position-based Gripper Action criterion: a gripper can report
  "closed" without actually holding anything, and effort is the signal that catches that case.

**Diagnosing a sim-success/real-failure case:** start with action-space or unit mismatches —
position vs. velocity control, radians vs. degrees, a flipped joint sign — checked simply by
plotting commanded vs. achieved joint trajectories from a single real rollout against sim. Next,
check the visual domain gap by running real camera frames through the policy's vision encoder to
see if embedding shifts point to lighting or texture issues rather than control problems. Then,
isolate latency or timing mismatches by replaying logged real observations through the policy
offline to see if the model's predicted actions were actually fine and just wrecked by
control-loop delays. Finally, tackle contact-dynamics issues like unmodeled friction or compliance
by tracking success rates phase-by-phase during grasp and contact, rather than just looking at
overall task completion.

### Egocentric

Egocentric video is what lets us actually verify the criteria above instead of just trusting
joint/gripper state — the arm's trajectory, and even the Grasp Force Confirmation signal, can look
identical whether the intended object was grasped, the wrong object was grasped, or the object
slipped mid-transport. Concretely, this changes evaluation in two ways:

1. **Failure diagnosis needs a video-review step specifically for the failure category "task
   looked kinematically/force-wise normal but the result was wrong."** For example, gripper effort
   stayed in-threshold for the whole transport phase (Grasp Force Confirmation would pass), but the
   camera shows it was holding the wrong object the whole time — a case the force signal alone
   cannot catch.
2. **Bonus — a simple egocentric success detector:** a lightweight binary classifier over the last
   few frames of `cam_left_wrist`/`cam_right_wrist` per rollout, trained on the episode-level
   success/failure labels from Task 2, as an automated screening signal for task completion. The
   useful design choice is to **cross-check it against Grasp Force Confirmation rather than trust
   either signal alone** — if the vision classifier and the effort-based criterion disagree (effort
   says holding, camera says empty or wrong object, or vice versa), that rollout gets flagged for
   mandatory human review instead of being auto-scored by whichever signal happened to run first.

---

## Task 5 — Model Adaptation (Bonus) — VLA

### Teleoperation

**Fine-tuning a pre-trained VLA (OpenVLA/pi0-style) on OpenArm data:** expected input format is
`(image(s), language instruction) -> action`, with action typically a normalized/discretized
version of the same 14-dim state-space seen throughout this audit (per-arm joint deltas or
absolute positions + gripper). The concrete adaptation work is mostly **action-space alignment**,
not architecture change: re-bin/re-normalize OpenArm's continuous 14-dim action into whatever
discretization scheme the pretrained VLA's action head expects (OpenVLA-style models typically use
per-dimension quantile binning fit on the fine-tuning dataset's own action distribution — which
means the real per-joint stats from `dataset_audit.py`'s outlier/range checks aren't just a Task 1
artifact, they're a direct input to this step, since binning against a distribution still
contaminated by outlier episodes silently wastes bin resolution on garbage values). Key
hyperparameters: learning rate (typically 1–2 orders of magnitude below pretraining LR — this is a
fine-tune, not training from scratch), LoRA rank if doing parameter-efficient fine-tuning rather
than full fine-tune (given OpenArm-scale data here — 50 episodes — full fine-tuning risks
catastrophic forgetting of the pretrained visual-language prior; LoRA is the safer default at this
data scale), and action-chunk length (how many future timesteps the model predicts per forward
pass — needs to match how the deployed control loop consumes actions).

### Egocentric

**Preprocessing/alignment for VLA fine-tuning:** resize/crop to the pretrained model's expected
resolution, then — given this audit's actual finding that 66% of `cam_left_wrist` frames are
meaningfully overexposed — a normalization step that's specifically robust to the exposure
distribution found here (e.g. per-frame histogram equalization or exposure-aware augmentation
during fine-tuning) matters more for this dataset than it would for a well-exposed third-person
benchmark; skipping it risks the VLA fitting to a lighting artifact rather than the task. Temporal
alignment with the action stream: sample the frame(s) closest to each action's timestamp within
the dataset's own tolerance window (LeRobot's own `tolerance_s` mechanism, already used implicitly
throughout this audit's data loading), not a separate hand-rolled alignment step.

**Failure modes when a third-person-pretrained VLA sees egocentric input at fine-tuning time:**
the model's pretrained visual prior was very likely built almost entirely on third-person
(over-the-shoulder or fixed workspace camera) data — egocentric wrist video has a categorically
different viewpoint distribution (close-range, camera co-moving with the gripper, frequent
partial self-occlusion by the gripper itself, as directly measured in this audit's occlusion-proxy
correlation). Concretely expect: (1) the pretrained vision encoder's features may not transfer
cleanly to close-range/motion-coupled imagery, effectively meaning the "fine-tune" has to relearn
a chunk of visual representation rather than lightly adapt one — arguing for a lower LoRA rank
being insufficient and a higher rank or partial vision-tower unfreeze being necessary specifically
for the egocentric camera stream; (2) the model may learn to (over-)rely on the third-person camera
stream, if the setup provides one, and effectively ignore the wrist camera, since gradient descent
will exploit whichever stream is easier to extract task-relevant signal from unless the training
setup (camera dropout during fine-tuning, e.g.) forces it not to; (3) given the real 66%
overexposure and blur-vs-motion finding from this audit, a frame-quality filter (this submission's
`egocentric_audit.py` blur/exposure detectors, exactly as used in `curation_pipeline.py`) applied
*before* fine-tuning data selection, not after, is what prevents the model from partly fitting to
image artifacts rather than task content.
