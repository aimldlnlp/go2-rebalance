"""Held-out cross-motor evaluation, reusing the frozen sudden-fault experiment."""
from project_paths import project_file

import argparse
import csv
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import sudden as s

OUT = s.ROOT / "results/cross_motor"
MOTORS = ["FR_calf", "RL_calf", "RR_calf"]


class CrossMotorBalance(s.SuddenBalance):
    def __init__(self, motor, **kwargs):
        if motor not in MOTORS:
            raise ValueError(f"Expected one of {MOTORS}")
        super().__init__(**kwargs)
        self.weak_motor = self.model.actuator(motor).id


def prepare():
    assert not OUT.exists(), "Refusing to replace an experiment"
    parent = s.verify_frozen()
    used, files = s.inventory()
    used.update(parent["evaluation_seeds"])
    for path in s.OUT.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(s.ROOT))] = s.digest(path)
    seeds = list(range(500, 520))
    assert not used.intersection(seeds)
    shared = ["checkpoints", "episode_s", "physics_dt_s", "control_dt_s", "measurement",
              "criteria", "recovery", "pairing", "execution", "failure_policy"]
    protocol = {key: parent[key] for key in shared}
    protocol.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                     "parent_protocol_sha256": s.digest(s.OUT / "protocol.json"),
                     "evaluation_seeds": seeds, "previously_used_seeds": sorted(used),
                     "motors": MOTORS, "controllers": ["pd", *parent["checkpoints"]],
                     "fault": {"onset_s": 5, "before": 1.0, "after": .1,
                               "schedule": "One selected motor per episode; 100% on [0,5), 10% on [5,30]. No ramp."},
                     "acceptance_scope": "Each of the three unchanged checkpoints must independently pass all three criteria on every tested motor. No pooling across motors. Any failed checkpoint/motor combination is reported and makes overall acceptance fail. No observation/action permutation, mirrored policy, retuning or retraining.",
                     "deliverables": "Protocol, configuration/checkpoint hashes, all 240 episode summaries, 500 Hz raw traces, tests and comparison report. No additional video or rendering."})
    OUT.mkdir(parents=True)
    s.write(OUT / "protocol.json", protocol)
    sources = ["cross_motor.py", "test_cross_motor.py", "sudden.py", "go2_env.py", "requirements-lock.txt"]
    s.write(OUT / "design.json", {"protocol_sha256": s.digest(OUT / "protocol.json"),
             "source_sha256": {name: s.digest(project_file(s.ROOT, name)) for name in sources},
             "prior_artifacts_sha256": files})
    print(f"Frozen: motors {MOTORS}, seeds 500-519, 240 episodes, {len(files)} prior artifacts protected", flush=True)


def verify():
    s.verify_frozen()
    design = s.read(OUT / "design.json")
    assert s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for name, sha in design["source_sha256"].items():
        assert s.digest(project_file(s.ROOT, name)) == sha, name
    p = s.read(OUT / "protocol.json")
    for checkpoint in p["checkpoints"].values():
        assert s.digest(s.ROOT / checkpoint["path"]) == checkpoint["sha256"]
    return p


def evaluate():
    p = verify()
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cpu", "cuda_build": torch.version.cuda, "torch": torch.__version__,
            "training": False, "rendering": False})
    policies = {"pd": None}
    for name, checkpoint in p["checkpoints"].items():
        policy = PPO.load(s.ROOT / checkpoint["path"], device="cpu")
        assert policy.num_timesteps == 262144
        assert all(torch.isfinite(parameter).all() for parameter in policy.policy.parameters())
        policies[name] = policy
    for motor in p["motors"]:
        folder = OUT / motor
        folder.mkdir()
        for name, policy in policies.items():
            kwargs = {} if policy is None else p["checkpoints"][name]["environment"]
            env = CrossMotorBalance(motor, **kwargs)
            assert env.model.actuator(env.weak_motor).name == motor
            rows = []
            with (folder / f"{name}_trace.csv").open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["seed", *s.FIELDS])
                for seed in p["evaluation_seeds"]:
                    started = time.perf_counter()
                    observation, _ = env.reset(seed=seed)
                    initial = s.digest_bytes(env.data.qpos.tobytes())
                    for _ in range(env.horizon):
                        action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                        observation, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
                    assert [r[0] for r in env.samples] == list(range(env.samples[-1][0] + 1))
                    writer.writerows((seed, *sample) for sample in env.samples)
                    handle.flush()
                    row = {"seed": seed, "motor": motor, "actuator_id": env.weak_motor,
                           "initial_state_sha256": initial, **s.summarize(env.samples),
                           "wall_s": time.perf_counter() - started}
                    s.write(folder / f"{name}_{seed}.json", row)
                    rows.append(row)
                    print(f"{motor}/{name} seed={seed}: post={row['mean_post_tilt_deg']:.3f} peak={row['max_tilt_deg']:.3f} time={row['simulated_s']:.3f} fallen={row['fallen']}", flush=True)
            env.close()
            s.write(folder / f"{name}.json", rows)
    audit()


def audit():
    p = verify()
    comparisons, counts, all_rows = {}, {}, {}
    for motor in p["motors"]:
        rows = {}
        for name in p["controllers"]:
            folder = OUT / motor
            recorded = s.read(folder / f"{name}.json")
            assert [r["seed"] for r in recorded] == p["evaluation_seeds"]
            raw = pd.read_csv(folder / f"{name}_trace.csv", float_precision="round_trip")
            assert sorted(raw.seed.unique()) == p["evaluation_seeds"]
            for row in recorded:
                assert row == s.read(folder / f"{name}_{row['seed']}.json")
                assert row["motor"] == motor
                group = raw.loc[raw.seed == row["seed"], s.FIELDS]
                np.testing.assert_array_equal(group.tick, np.arange(len(group)))
                np.testing.assert_array_equal(group.time_s, group.tick * s.DT)
                np.testing.assert_array_equal(group.strength_next_step, np.where(group.tick < s.ONSET, 1, .1))
                np.testing.assert_array_equal(group.strength_applied, np.where(group.tick <= s.ONSET, 1, .1))
                summary = s.summarize(group.to_numpy(dtype=float))
                assert summary == {key: row[key] for key in summary}
                # Independent group reductions also check the reused summary implementation.
                np.testing.assert_allclose(group.loc[group.tick > s.ONSET].tilt_deg.mean(), row["mean_post_tilt_deg"], rtol=0, atol=1e-12)
                assert group.tilt_deg.max() == row["max_tilt_deg"]
            rows[name] = recorded
            counts[f"{motor}/{name}"] = len(raw)
        comparisons[motor] = {name: s.criteria(rows["pd"], rows[name]) for name in p["checkpoints"]}
        all_rows[motor] = rows
    for name in p["controllers"]:
        assert len({tuple(r["initial_state_sha256"] for r in all_rows[motor][name]) for motor in p["motors"]}) == 1
    result = {"passed": all(c["passed"] for motor in comparisons.values() for c in motor.values()),
              "motors": comparisons, "raw_sample_counts": counts,
              "paired_initial_states_verified": True, "protocol_sha256": s.digest(OUT / "protocol.json")}
    target = OUT / "acceptance.json"
    if target.exists():
        assert s.read(target) == result
    else:
        s.write(target, result)
    design = s.read(OUT / "design.json")
    for path, sha in design["prior_artifacts_sha256"].items():
        assert s.digest(s.ROOT / path) == sha, path
    execution = s.read(OUT / "execution.json")
    assert datetime.fromisoformat(p["created_utc"]) < datetime.fromisoformat(execution["started_utc"])
    log = (OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "Ran 23 tests" in log and "OK" in log and "FAILED" not in log
    table = []
    for motor, controllers in comparisons.items():
        for name, c in controllers.items():
            survived = sum(not r["fallen"] and r["simulated_s"] == 30 for r in all_rows[motor][name])
            peaks = sum(pair["peak_passed"] for pair in c["pairs"])
            table.append(f"| {motor} | {name} | {c['pd_mean_post_tilt_deg']:.4f} | {c['ppo_mean_post_tilt_deg']:.4f} | {c['reduction_pct']:.2f}% | {survived}/20 | {peaks}/20 | {'PASS' if c['passed'] else 'FAIL'} |")
    report = f"""# Cross-motor generalization

**Overall acceptance: {'PASS' if result['passed'] else 'FAIL'}.** Existing policies were tested without retraining, action mirroring, or changing observations. Each run weakens one previously untested knee motor from full strength to 10% at 5 seconds. All three learned checkpoints are assessed separately for every motor.

| Faulted motor | Checkpoint | PD post-fault mean (°) | PPO post-fault mean (°) | Reduction | Survived 30 s | Paired peak checks | Result |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

FR_calf is the front-right knee, RL_calf the rear-left knee, and RR_calf the rear-right knee. Checkpoint labels refer to training seeds 0, 1, and 2. The policies were trained on weakness of FL_calf, the front-left knee. No result from the new motors was used to select a checkpoint or tune the controller.

## Frozen protocol

`protocol.json` was saved at {p['created_utc']}, before evaluation began at {execution['started_utc']}. Seeds 500–519 are new relative to the stored prior seed inventory, including the previous sudden-fault seeds 400–419. The same 20 seeds are used for all motors and controllers, with identical initial-state hashes. Each motor is faulted independently; no simultaneous failures are introduced.

Acceptance is unchanged: every episode must complete 30 seconds without falling; each checkpoint's mean post-fault tilt across 20 seeds must be at least 40% below paired PD; every individual paired peak must be no larger than PD. Each motor/checkpoint combination must pass all three checks. A pass elsewhere does not compensate for a failure. `acceptance.json` preserves all individual pair checks, including failures.

The inherited controller and simulator are unchanged. PD gains are 50 and 3. The learned joint-target residual has scale 0.25 rad, a 2° deadband, and a one-degree blend. Predictions are deterministic. Only the selected actuator's clipped torque is multiplied by the fault strength. Physics runs at 500 Hz and the policy at 50 Hz. At 5 seconds the multiplier changes for the next integration interval; the first affected state sample is at 5.002 seconds.

Mean tilt uses samples in (5,30] seconds, or until termination if a fall occurs. Shortened episodes always fail survival; their means are not evidence of equal-duration performance. Peaks cover the entire observed episode. Falls mean base height below 0.16 m, tilt above 45°, or a nonfinite state. Recovery uses the previous frozen definition: reach the [4,5]-second mean plus 0.5° for one continuous second. It is diagnostic only; per-episode recovery times and null values for non-recovery are preserved in JSON.

## Evidence

`protocol.json` and `design.json` freeze criteria, seed choices, configuration, checkpoint hashes and source hashes. Each motor folder contains PD and three learned-controller JSON summaries, individual episode JSON files, and matching 500 Hz `*_trace.csv` files. `execution.json` and `evaluation.log` record actual execution. All 240 requested episodes were run. Raw traces contain {sum(counts.values()):,} samples, including initial states. All summaries were recomputed from the saved traces. The 23 functional and regression tests passed before evaluation; see `tests-before.log`.

All {len(design['prior_artifacts_sha256'])} protected prior artifacts and all three checkpoints retain their hashes. `completion_audit.json` records final evidence checks. No training, rendering or CUDA ran, and no dependency was added. No new video was produced for this experiment.

Recheck existing evidence from the project directory:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest test_cross_motor test_sudden test_go2 test_improve -v
.\\.venv\\Scripts\\python.exe cross_motor.py audit
```

Preparation and evaluation refuse to overwrite an existing experiment. This tests single knee-motor gain loss on a flat simulated floor with small initial joint-position variations. It does not establish walking performance, other joint failures, simultaneous faults, rough-ground behavior, or transfer to physical hardware. Sampled peaks do not bound motion between physics samples.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    proof = {"completed": True, "experimental_acceptance_passed": result["passed"],
             "episodes": 240, "raw_samples": sum(counts.values()), "tests_passed_before_evaluation": 23,
             "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]),
             "checkpoints_and_sources_unchanged": True, "paired_initial_states_verified": True,
             "all_raw_summaries_verified": True, "training": False, "rendering": False, "cuda": False}
    target = OUT / "completion_audit.json"
    if target.exists():
        assert s.read(target) == proof
    else:
        s.write(target, proof)
    print("\n".join(table), flush=True)
    print("Overall acceptance:", result["passed"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate", "audit"])
    {"prepare": prepare, "evaluate": evaluate, "audit": audit}[parser.parse_args().stage]()
