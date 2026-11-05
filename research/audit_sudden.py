"""Check persisted experiment evidence and write the final comparison report."""

from contextlib import redirect_stdout
from datetime import datetime
import io

import numpy as np

import sudden as s


def main():
    with redirect_stdout(io.StringIO()):
        result = s.audit()
    p = s.verify_frozen()
    design = s.read(s.OUT / "design.json")
    for path, sha in design["prior_artifacts_sha256"].items():
        assert s.digest(s.ROOT / path) == sha, path
    execution = s.read(s.OUT / "execution.json")
    assert datetime.fromisoformat(p["created_utc"]) < datetime.fromisoformat(execution["started_utc"])
    assert execution["device"] == "cpu" and execution["cuda_build"] is None and not execution["training"]
    tests = (s.OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "Ran 20 tests" in tests and "OK" in tests and "FAILED" not in tests
    video = s.read(s.OUT / "video_verification.json")
    assert video["passed"] and video["seed"] == p["video_seed"] == 400
    assert video["controllers"] == p["video_controllers"]
    assert video["all_frames_decoded"] and video["decoded_frames"] == 750
    assert video["seconds"] == 30 and video["fps"] == 25 and video["size"] == [1920, 1080]
    assert s.digest(s.OUT / "sudden_comparison_seed_400.mp4") == video["video_sha256"]
    motion = np.load(s.OUT / "video_motion.npz")
    rows = {name: s.read(s.OUT / f"{name}.json") for name in p["video_controllers"]}
    for name in rows:
        row = next(r for r in rows[name] if r["seed"] == 400)
        assert s.digest_bytes(motion[name + "_qpos"][0].tobytes()) == row["initial_state_sha256"]
        assert len(motion[name + "_qpos"]) == len(motion[name + "_tilt"]) == 1501
    assert all(n == 300020 for n in result["raw_sample_counts"].values())
    pd = rows["pd"]
    table = []
    for name, group in rows.items():
        survived = sum(not r["fallen"] and r["simulated_s"] == 30 for r in group)
        mean = float(np.mean([r["mean_post_tilt_deg"] for r in group]))
        peak = max(r["max_tilt_deg"] for r in group)
        recovered = sum(r["recovered"] for r in group)
        if name == "pd":
            table.append(f"| PD | {mean:.4f} | Reference | {peak:.4f} | Reference | {survived}/20 | {recovered}/20 | Reference |")
        else:
            c = result["checkpoints"][name]
            peak_pass = sum(r["peak_passed"] for r in c["pairs"])
            table.append(f"| PD + PPO, training seed {p['checkpoints'][name]['training_seed']} | {mean:.4f} | {c['reduction_pct']:.2f}% | {peak:.4f} | {peak_pass}/20 | {survived}/20 | {recovered}/20 | {'PASS' if c['passed'] else 'FAIL'} |")
    mean_pd = result["checkpoints"]["ppo_seed_0"]["pd_mean_post_tilt_deg"]
    total_wall = sum(r["wall_s"] for group in rows.values() for r in group)
    report = f"""# Sudden motor-strength loss: held-out evaluation

**Overall result: {'PASS' if result['passed'] else 'FAIL'}.** Three existing checkpoints were tested without further training on 20 new evaluation seeds. The front-left knee motor changed directly from 100% to 10% strength at 5 seconds. Every result below comes from this fixed experiment.

| Controller | Mean tilt after fault (°) | Reduction vs PD | Largest measured tilt (°) | Paired peak checks | Completed without falling | Recovered | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

The mean criterion is evaluated independently for each checkpoint across its 20 matched episodes. PD's post-fault mean is {mean_pd:.6f}°, so each learned checkpoint must stay at or below {mean_pd * .6:.6f}°. The peak criterion is stricter at the episode level: every learned run must have a peak no larger than its paired PD run. A good average cannot compensate for a failed pair. Individual pairs, including failures, are in `acceptance.json`.

## Protocol and fixed conditions

The protocol was written at {p['created_utc']}, before evaluation began at {execution['started_utc']}. Its SHA-256 is `{s.digest(s.OUT / 'protocol.json')}`. Checkpoint paths, checkpoint hashes, controller settings, seed choices, thresholds, recovery definition, and the video example were fixed before the test. No checkpoint was changed and no search or training took place.

Evaluation seeds were 400–419. The recorded previous seed inventory includes training seeds 0–2, validation seeds 100–104, and previous tests 200–209 and 300–309. The new seeds have no intersection with that inventory. Tests use seed 987654 for the onset check, outside the evaluation set.

PD is computed with gains 50 and 3 at each 0.002-second physics step. PPO runs deterministically every 0.02 seconds and corrects joint targets, with the already selected ±0.25 rad residual scale and 2° deadband blended over the next degree. The body, physical model, joint limits, initial joint noise of ±0.01 rad, and fall definition are unchanged. A fall means base height below 0.16 m, body tilt above 45°, or a nonfinite state.

At integer physics tick 2500 (5 seconds), the motor multiplier changes to 0.1 for the next integration interval. The sample at exactly 5 seconds still describes motion produced by full strength. The first affected integration sample is at 5.002 seconds. `strength_applied` and `strength_next_step` in the raw traces make this boundary explicit. All other motors retain full strength. Initial-state hashes match for PD and each learned controller on every seed.

## Measurements and recovery

Tilt is measured with the existing environment's body orientation metric, at every physics step, including the healthy phase. The primary mean uses exactly 12,500 samples per completed episode over (5,30] seconds, followed by an equal-weight mean across the 20 episodes. Peak checks use the full [0,30] interval sampled at 500 Hz; post-fault-only peaks are saved as an additional diagnostic. This differs from the earlier gradual-fault report, which used whole-episode averages at the control rate. Those older numbers are not directly interchangeable with these means.

Recovery means reaching the controller's own pre-fault mean over [4,5] seconds plus 0.5°, then remaining at or below that bound for at least one continuous second. The elapsed time is measured from the fault to the beginning of the first qualifying interval. An episode that never leaves the bound is recorded as zero; no qualifying interval is recorded as null, right-censored at the episode's end. Recovery is diagnostic and does not add or replace an acceptance criterion. Per-episode bounds and recovery times are in the controller JSON files. Reduced tilt does not by itself demonstrate a return to the pre-fault posture.

## Video

`sudden_comparison_seed_400.mp4` shows the preregistered seed 400, with PD and all three learned checkpoints in the same silent 30-second video. The model labels 1, 2, and 3 correspond to training seeds 0, 1, and 2. Poses were captured during evaluation; no replay was selected after seeing the results. Every recorded control-rate tilt used for the video matches the corresponding 500 Hz raw trace. The video uses 750 frames at 25 frames per second, 1920 × 1080 resolution, and English labels. Native simulator rendering changes only the presentation model's colors and camera; the evaluation itself is headless. The font file and video hash are recorded in `video_verification.json`.

## Evidence and reproduction

- `protocol.json`, `design.json`: preregistration, selected checkpoints and hashes, protected source and prior-artifact hashes.
- `execution.json`, `evaluation.log`: inference-only execution metadata and all episode outcomes.
- `pd.json`, `ppo_seed_0.json`, `ppo_seed_1.json`, `ppo_seed_2.json`: all 80 episode summaries, including recovery times and initial hashes.
- Matching `*_trace.csv` files: 300,020 rows per controller, including initial states, for 1,200,080 total physics samples.
- `acceptance.json`: unchanged aggregate mean and per-pair peak checks, with no tolerance added to tilt thresholds.
- `tests-before.log`: all 20 functional and regression tests passed before evaluation.
- `video_motion.npz`, `video_verification.json`, `video_contact_sheet.png`: recorded poses, decoded-video checks, and visual review samples.
- `completion_audit.json`: final evidence checks, checkpoint integrity, prior-artifact preservation, and artifact hashes.

Commands from the project directory:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest test_sudden test_go2 test_improve -v
.\\.venv\\Scripts\\python.exe sudden.py audit
```

The original run was `sudden.py prepare`, tests, `sudden.py evaluate`, `sudden_video.py`, then `audit_sudden.py`. Preparation and evaluation deliberately refuse overwriting an existing experiment. Evaluation episode work took {total_wall:.2f} seconds in total, excluding startup, checkpoint loading, final audit, and video rendering. Training did not run. PyTorch is `{execution['torch']}`, with no CUDA build.

## Limits

This tests one faulted motor, one sudden strength change, small variations in starting joint position, and a flat floor in simulation. It does not establish walking performance, other motor failures, mechanical damage, real motor behavior, or transfer to physical hardware. A 500 Hz sampled peak does not bound a continuous-time peak between physics samples. Recovery is judged over a 30-second observation window; null values must not be read as fast recovery. Video examples illustrate one seed and do not replace the full 20-seed comparison.
"""
    (s.OUT / "REPORT.md").write_text(report, encoding="utf-8")
    hashes = {str(path.relative_to(s.OUT)): s.digest(path) for path in sorted(s.OUT.iterdir())
              if path.is_file() and path.name != "completion_audit.json"}
    s.write(s.OUT / "completion_audit.json", {
        "completed": True, "experimental_acceptance_passed": result["passed"],
        "protocol_precedes_evaluation": True, "new_eval_seeds": p["evaluation_seeds"],
        "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]),
        "checkpoints_unchanged": True, "frozen_sources_unchanged": True,
        "paired_initial_states_verified": True, "all_raw_summaries_recomputed": True,
        "raw_physics_rows": sum(result["raw_sample_counts"].values()),
        "tests_passed_before_evaluation": 20, "training": False, "cuda": False,
        "video_seed_preselected": 400, "video_verified": True,
        "artifact_sha256": hashes, "audit_source_sha256": s.digest(__file__)})
    print("Completion audit passed. Experiment acceptance:", result["passed"])
    print("\n".join(table))


if __name__ == "__main__":
    main()
