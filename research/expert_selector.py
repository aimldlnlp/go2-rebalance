"""Frozen inference-only expert selector on five fresh diagnostic starts."""
from project_paths import project_file

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import time

import numpy as np
import torch
from stable_baselines3 import PPO

import multimotor as m
import multimotor_extend as extension
import review_multimotor as review

OUT = m.s.ROOT / "results/expert_selector"
MAPPING = {"healthy": None, "FL_calf": 2, "FR_calf": 0, "RL_calf": 2, "RR_calf": 1}
SEEDS = list(range(800, 805))


def prepare():
    assert not OUT.exists(), "Refusing to overwrite an experiment"
    parent = extension.verify()
    analysis_path = m.s.ROOT / "results/next_step_analysis/analysis.json"
    analysis = m.s.read(analysis_path)
    assert {c: r["training_seed"] for c, r in analysis["post_hoc_expert_selection"].items()} == {
        c: seed for c, seed in MAPPING.items() if seed is not None}
    for name, sha in analysis["source_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    used, protected = m.s.inventory()
    used.update(m.s.read(m.s.OUT / "protocol.json")["evaluation_seeds"])
    for path in m.s.OUT.rglob("*"):
        if path.is_file():
            protected[str(path.relative_to(m.s.ROOT))] = m.s.digest(path)
    assert not set(SEEDS).intersection(used)
    assert not set(SEEDS).intersection(parent["final_seeds"])
    checkpoints = {}
    manifest = m.s.read(extension.OUT / "evidence_manifest.json")
    for seed in range(3):
        path = extension.OUT / f"round_1048576/seed_{seed}/policy.zip"
        config = m.s.read(path.parent / "config.json")
        assert config["environment"] == parent["environment"] and config["target_total_steps"] == 1048576
        sha = m.s.digest(path)
        assert sha == manifest[str(path.relative_to(extension.OUT))]
        checkpoints[str(seed)] = {"path": str(path.relative_to(m.s.ROOT)), "sha256": sha}
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_seeds": SEEDS, "prior_recorded_seeds": sorted(used),
        "reserved_final_seeds": parent["final_seeds"], "retired_diagnostic_seeds": SEEDS,
        "conditions": m.CONDITIONS, "mapping": MAPPING, "checkpoints": checkpoints,
        "environment": parent["environment"], "criteria": parent["criteria"],
        "measurement": parent["measurement"], "episodes": 50,
        "fault": {"onset_s": 5, "before": 1.0, "after": .1, "schedule": "Single knee, step change, no ramp"},
        "selector": "Simulator scenario identity selects one frozen expert for the ENTIRE episode; healthy uses PD with zero residual. No online switching, action averaging, learned fault detection, or extra policy observations.",
        "selection_basis": "Retrospective validation 100-104; all experts at 1048576 steps. Mapping fixed before any fresh diagnostic replay.",
        "analysis_sha256": m.s.digest(analysis_path),
        "execution": {"device": "cpu", "torch_threads": 1, "training": False, "rendering": False, "cuda": False},
        "failure_policy": "Keep all results, no remapping, tuning, second seed batch, or threshold changes after results. Diagnostic seeds are retired from future final evaluation.",
        "interpretation": "Diagnostic only. Privileged motor identity; not one unprivileged policy or three independently passing policies. Passing permits investigating teacher transfer; failing does not establish reliable teachers."}
    sources = ["expert_selector.py", "test_expert_selector.py", "multimotor.py", "multimotor_extend.py",
               "go2_env.py", "sudden.py", "review_multimotor.py", "requirements-lock.txt"]
    source_hashes = {name: m.s.digest(project_file(m.s.ROOT, name)) for name in sources}
    model_hashes = {str(path.relative_to(m.s.ROOT)): m.s.digest(path)
                    for path in (m.s.ROOT / "vendor/mujoco_menagerie/unitree_go2").rglob("*") if path.is_file()}
    assert model_hashes
    OUT.mkdir()
    m.s.write(OUT / "protocol.json", protocol)
    m.s.write(OUT / "design.json", {"protocol_sha256": m.s.digest(OUT / "protocol.json"),
        "source_sha256": source_hashes, "model_sha256": model_hashes, "prior_artifacts_sha256": protected})
    print(f"Frozen: seeds {SEEDS}, mapping {MAPPING}, 50 episodes; {len(protected)} prior artifacts protected", flush=True)


def verify():
    design = m.s.read(OUT / "design.json")
    assert m.s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for name, sha in {**design["source_sha256"], **design["model_sha256"]}.items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    p = m.s.read(OUT / "protocol.json")
    assert p["mapping"] == MAPPING and p["diagnostic_seeds"] == SEEDS
    assert not set(SEEDS).intersection(p["prior_recorded_seeds"] + p["reserved_final_seeds"])
    assert p["environment"] == {"residual_scale": .35, "residual_penalty": .03,
                                "tilt_deadband_deg": 2., "tilt_weight": 5.}
    assert p["criteria"] == m.verify()["criteria"]
    for checkpoint in p["checkpoints"].values():
        assert m.s.digest(m.s.ROOT / checkpoint["path"]) == checkpoint["sha256"]
    return p


def evaluate(policies, folder, p):
    folder.mkdir()
    rows = []
    with (folder / "trace.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", *m.s.FIELDS])
        for condition in p["conditions"]:
            policy = policies[condition]
            env = m.MultiMotorBalance(motor=condition, record=True, **p["environment"])
            try:
                for seed in p["diagnostic_seeds"]:
                    observation, _ = env.reset(seed=seed)
                    initial = m.s.digest_bytes(env.data.qpos.tobytes())
                    full_initial = m.s.digest_bytes(b"".join(a.tobytes() for a in
                        (env.data.qpos, env.data.qvel, env.data.ctrl, observation)))
                    for _ in range(env.horizon):
                        action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                        observation, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
                    writer.writerows((condition, seed, *sample) for sample in env.samples)
                    handle.flush()
                    row = {"condition": condition, "seed": seed, "initial_state_sha256": initial,
                        "full_initial_state_sha256": full_initial, **m.s.summarize(env.samples),
                        "mean_tilt_deg": float(np.mean([r[2] for r in env.samples[1:]]))}
                    rows.append(row)
                    print(f"{folder.name}/{condition}/{seed}: post={row['mean_post_tilt_deg']:.4f}, peak={row['max_tilt_deg']:.4f}, seconds={row['simulated_s']}, fallen={row['fallen']}", flush=True)
            finally:
                env.close()
    m.s.write(folder / "episodes.json", rows)
    return rows


def run():
    assert not (OUT / "execution.json").exists(), "Refusing to rerun diagnostics"
    p = verify()
    tests = (OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "\nOK" in tests and "FAILED" not in tests
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    policies = {}
    for seed, checkpoint in p["checkpoints"].items():
        policy = PPO.load(m.s.ROOT / checkpoint["path"], device="cpu")
        assert policy.num_timesteps == 1048576 and policy.seed == int(seed)
        assert all(torch.isfinite(parameter).all() for parameter in policy.policy.parameters())
        policies[int(seed)] = policy
    m.s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "training": False, "rendering": False, "cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(), "tests_sha256": m.s.digest(OUT / "tests-before.log"),
        "versions": {name: version(name) for name in ["torch", "mujoco", "numpy", "stable-baselines3", "gymnasium"]}})
    started = time.perf_counter()
    baseline = evaluate({c: None for c in p["conditions"]}, OUT / "pd", p)
    selector = evaluate({c: None if seed is None else policies[seed] for c, seed in p["mapping"].items()}, OUT / "selector", p)
    m.s.write(OUT / "acceptance.json", m.evidence(baseline, selector, p["diagnostic_seeds"]))
    m.s.write(OUT / "timing.json", {"evaluation_wall_s": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat()})
    audit()


def audit():
    p = verify()
    execution = m.s.read(OUT / "execution.json")
    assert datetime.fromisoformat(p["created_utc"]) < datetime.fromisoformat(execution["started_utc"])
    assert execution["tests_sha256"] == m.s.digest(OUT / "tests-before.log")
    baseline, n_pd = review.verify_dataset(OUT / "pd", SEEDS)
    selector, n_selector = review.verify_dataset(OUT / "selector", SEEDS)
    for before, after in zip(baseline, selector, strict=True):
        assert (before["condition"], before["seed"]) == (after["condition"], after["seed"])
        assert before["full_initial_state_sha256"] == after["full_initial_state_sha256"]
    healthy_pd = [r for r in baseline if r["condition"] == "healthy"]
    healthy_selector = [r for r in selector if r["condition"] == "healthy"]
    assert healthy_pd == healthy_selector
    result = m.s.read(OUT / "acceptance.json")
    independent = review.independent_comparison(baseline, selector, SEEDS)
    review.assert_comparison(independent, result)
    design = m.s.read(OUT / "design.json")
    for name, sha in design["prior_artifacts_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    proof = {"completed": True, "diagnostic_passed": result["passed"], "episodes": len(baseline) + len(selector),
        "raw_samples": n_pd + n_selector, "independent_raw_metrics_verified": True,
        "full_paired_initial_states_verified": True, "healthy_equals_pd": True,
        "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]),
        "checkpoint_source_model_hashes_verified": True, "reserved_final_seeds_used": False,
        "training": False, "rendering": False, "cuda": False, "independent_comparison": independent}
    assert proof["episodes"] == p["episodes"] == 50
    table = []
    for group in independent["groups"]:
        c = group["condition"]
        passed = (independent["checks"]["healthy"] if c == "healthy" else
                  independent["checks"][c + "_mean"] and independent["checks"][c + "_peak"])
        passed = passed and group["survived"] == 5
        table.append(f"| {c} | {p['mapping'][c] if c != 'healthy' else 'PD'} | {group['pd_mean_deg']:.4f} | {group['ppo_mean_deg']:.4f} | {group['reduction_pct']:.2f}% | {group['survived']}/5 | {group['peak_passes']}/5 | {'PASS' if passed else 'FAIL'} |")
    report = f"""# Fixed expert selector: fresh diagnostic starts

**Diagnostic result: {'PASS' if result['passed'] else 'FAIL'}.** This is a privileged selector, not acceptance of one policy or three independent policies.

| Condition | Expert training seed | PD mean (degrees) | Selector mean (degrees) | Reduction | Survived 30 s | Paired peaks | Result |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

## Frozen experiment

Mapping: FL/RL -> seed 2, FR -> seed 0, RR -> seed 1, healthy -> PD. All checkpoints have 1,048,576 steps; residual scale remains 0.35 rad. Scenario identity comes from the simulator and selects the expert for the entire episode, including the initial healthy phase. No online switching, action averaging, fault detector, or new policy observations. No training, CUDA or rendering.

Protocol frozen at {p['created_utc']}; execution started at {execution['started_utc']}. Diagnostic seeds 800-804 were checked against recorded JSON/CSV seed inventory before freezing. These seeds are retired from future final evaluation. Reserved final seeds 700-719 were not run. All 50 episodes (5 seeds x 5 conditions x 2 controllers) and failures are retained. No remapping or second diagnostic batch after results.

Dynamics unchanged: 500 Hz physics, 50 Hz deterministic control, 30 seconds; single knee changes from 100% to 10% at 5 seconds. Healthy means use (0,30]; fault means use (5,30], equal weight across five seeds per condition. Peaks cover the full episode. Every episode must survive; healthy mean <= PD + 0.3 degrees; each fault mean <= 60% of PD; every fault paired peak <= PD. A pass elsewhere cannot offset a failure. Recovery is diagnostic only.

## Audit and evidence

`protocol.json` and `design.json` freeze mapping, checkpoint/config/source/model hashes, prior seed inventory and thresholds. `tests-before.log` records regression checks before execution. `execution.json` records CPU/runtime versions. `pd/` and `selector/` contain every episode summary and raw 500 Hz `trace.csv`. `acceptance.json` retains all checks and peak pairs; `completion_audit.json` contains independently recomputed means, peaks, recovery and pass/fail checks. Full initial qpos/qvel/ctrl/observation hashes agree for every pair; healthy episode summaries are identical.

All {n_pd + n_selector:,} raw samples were checked for condition/seed coverage, contiguous ticks, fault strength boundaries, finite values, duration, fall flags, means, peaks and recovery. All {proof['prior_artifacts_unchanged']} prior artifacts and all frozen checkpoint/source/model hashes remain unchanged. Protocol timestamps precede execution; test log hash is preserved. No final evaluation directory was created.

## Interpretation and next decision

{'The fixed expert collection passes this small fresh-start diagnostic. Investigating transfer to one policy using ordinary observations is justified, but transfer has not been run or proven.' if result['passed'] else 'The fixed expert collection fails this fresh-start diagnostic. Do not treat these selected experts as reliable teachers; inspect the failed conditions before another training proposal. Mapping and failures remain frozen.'}

Only five starts with small reset noise, fixed fault strength/onset, flat floor, and perfect scenario identity were tested. This cannot isolate observation versus optimization causes, establish broad generalization, satisfy three independently trained policies, or demonstrate hardware readiness. Post-hoc old-validation selection remains the basis for the mapping; only this new diagnostic set is fresh.

Recheck: `.venv\\Scripts\\python.exe expert_selector.py audit`. Prepare/run refuse overwriting.
"""
    for name, value in [("completion_audit.json", proof)]:
        path = OUT / name
        if path.exists():
            assert m.s.read(path) == value
        else:
            m.s.write(path, value)
    report_path = OUT / "REPORT.md"
    if report_path.exists():
        assert report_path.read_text(encoding="utf-8") == report
    else:
        with report_path.open("x", encoding="utf-8") as handle:
            handle.write(report)
    print("\n".join(table), flush=True)
    print(f"Independent audit complete: diagnostic_passed={result['passed']}, samples={n_pd+n_selector}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "audit"])
    {"prepare": prepare, "run": run, "audit": audit}[parser.parse_args().stage]()
