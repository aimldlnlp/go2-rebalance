"""Continue configuration B at two fixed budgets; select only on old validation seeds."""
from project_paths import project_file

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import multimotor as m
import review_multimotor as review

OUT = m.s.ROOT / "results/multimotor_extended"


def initial_parents():
    return {0: m.OUT / "candidates/B", 1: m.OUT / "replicates/seed_1", 2: m.OUT / "replicates/seed_2"}


def prepare():
    assert not OUT.exists(), "Refusing to overwrite an experiment"
    original = m.verify()
    assert m.s.read(m.OUT / "selection.json")["selected"] == "B"
    manifest = m.s.read(m.OUT / "evidence_manifest.json")
    for path, sha in manifest.items():
        assert m.s.digest(m.OUT / path) == sha, path
    used, files = m.s.inventory()
    used.update(m.s.read(m.s.OUT / "protocol.json")["evaluation_seeds"])
    for path in m.s.OUT.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(m.s.ROOT))] = m.s.digest(path)
    final = list(range(700, 720))
    assert not set(final).intersection(used)
    parents = {str(seed): {"path": str(path.relative_to(m.s.ROOT)),
                           "sha256": m.s.digest(path / "policy.zip")} for seed, path in initial_parents().items()}
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": "B",
        "environment": original["configurations"]["B"], "ppo": original["ppo"],
        "initial_total_steps": 262144, "total_step_checkpoints": [524288, 1048576],
        "training_seeds": [0, 1, 2], "validation_seeds": original["validation_seeds"],
        "final_seeds": final, "retired_final_seeds": original["final_seeds"], "prior_used_seeds": sorted(used),
        "conditions": m.CONDITIONS, "criteria": original["criteria"], "measurement": original["measurement"],
        "training_distribution": original["training_distribution"].removeprefix("From scratch; "), "parents": parents,
        "continuation": "Restore each seed's own policy, Adam optimizer and timestep count from its preceding checkpoint. Restart global and environment random generators with the original training seed at each continuation stage; the earlier RNG/rollout stream was not saved, so this is not bit-exact restoration of uninterrupted training. Do not reset weights, change hyperparameters, alter the controller/reward, or use other configurations.",
        "selection": "After each fixed total-step budget, evaluate all three seeds on validation 100-104 using the unchanged stored paired PD reference. Independently verify all raw validation traces. Select the earliest budget at which every criterion passes independently for all three seeds and all four motors. Stop further training immediately after common success. Maximum total steps: 1048576 per seed.",
        "final_evaluation": "Only after common validation success, close training and freeze the three checkpoint hashes. Run one evaluation on new seeds 700-719: healthy plus four separate knee faults, PD versus each checkpoint. No selection, tuning or further training based on final results. If neither budget passes validation, do not create a final test or use the reserved final seeds.",
        "failure_policy": "Keep every checkpoint, validation failure and raw trace. Do not loosen healthy +0.3 degrees, 40% fault mean reduction, 30-second no-fall requirement, or per-pair peak <= PD. Do not pool between motors or training seeds.",
        "execution": {"device": "cpu", "torch_threads": 1, "cuda": False, "rendering": False},
        "deliverables": "New protocol/configuration/source hashes, optimizer continuation evidence, checkpoints, monitor logs, raw validation data and any permitted final data, tests, independent calculation audit, comparison figure and Markdown report. Preserve all earlier artifacts; no new video."}
    OUT.mkdir(parents=True)
    m.s.write(OUT / "protocol.json", protocol)
    sources = ["multimotor_extend.py", "test_multimotor_extend.py", "multimotor.py", "review_multimotor.py",
               "go2_env.py", "sudden.py", "requirements-lock.txt"]
    m.s.write(OUT / "design.json", {"protocol_sha256": m.s.digest(OUT / "protocol.json"),
        "source_sha256": {name: m.s.digest(project_file(m.s.ROOT, name)) for name in sources},
        "prior_artifacts_sha256": files})
    print(f"Protocol frozen: B; totals 524288/1048576; validation 100-104; reserved final 700-719; {len(files)} old artifacts protected", flush=True)


def verify():
    m.verify()
    p, design = m.s.read(OUT / "protocol.json"), m.s.read(OUT / "design.json")
    assert m.s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for name, sha in design["source_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    for item in p["parents"].values():
        assert m.s.digest(m.s.ROOT / item["path"] / "policy.zip") == item["sha256"]
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def resume(parent, seed, previous, monitor):
    agent = PPO.load(parent / "policy.zip", env=monitor, device="cpu")
    if agent.seed != seed or agent.num_timesteps != previous:
        raise ValueError("Parent seed or timestep count does not match")
    assert agent.policy.optimizer.state
    agent.set_random_seed(seed)
    return agent


def optimizer_steps(agent):
    return int(next(iter(agent.policy.optimizer.state.values()))["step"].item())


def continue_seed(parent, previous, total, seed, p):
    directory = OUT / f"round_{total}/seed_{seed}"
    directory.mkdir(parents=True)
    parent_hash = m.s.digest(parent / "policy.zip")
    m.s.write(directory / "config.json", {"configuration": "B", "training_seed": seed,
        "environment": p["environment"], "ppo": p["ppo"], "previous_total_steps": previous,
        "target_total_steps": total, "parent_checkpoint_sha256": parent_hash,
        "protocol_sha256": m.s.digest(OUT / "protocol.json"), "rng_restart_at_stage": True})
    env = m.MultiMotorBalance(randomize=True, **p["environment"])
    monitor = Monitor(env, str(directory / "monitor.csv"), info_keywords=("fault_motor", "final_strength", "fault_onset_s"))
    agent = resume(parent, seed, previous, monitor)
    before = {name: parameter.detach().clone() for name, parameter in agent.policy.named_parameters()}
    steps_before = optimizer_steps(agent)
    started = time.perf_counter()
    agent.learn(total_timesteps=total - previous, reset_num_timesteps=False)
    assert agent.num_timesteps == total and optimizer_steps(agent) > steps_before
    assert all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters())
    assert any(not torch.equal(before[name], parameter) for name, parameter in agent.policy.named_parameters())
    agent.save(directory / "policy")
    restored = PPO.load(directory / "policy", device="cpu")
    check = m.MultiMotorBalance(**p["environment"])
    observation, _ = check.reset(seed=100)
    np.testing.assert_array_equal(agent.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
    check.close()
    monitor.close()
    memory = psutil.Process().memory_info()
    assert m.s.digest(parent / "policy.zip") == parent_hash
    m.s.write(directory / "training.json", {"actual_total_steps": total, "additional_steps": total - previous,
        "wall_s": time.perf_counter() - started, "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
        "optimizer_steps_before": steps_before, "optimizer_steps_after": optimizer_steps(agent),
        "policy_parameters_changed": True, "reload_predictions_identical": True,
        "checkpoint_sha256": m.s.digest(directory / "policy.zip"), "parent_checkpoint_sha256": parent_hash})
    print(f"Training complete: total={total}, seed={seed}", flush=True)
    rows = m.evaluate(restored, directory / "validation", p["validation_seeds"], p["environment"])
    baseline = m.s.read(m.OUT / "validation/pd/episodes.json")
    result = m.evidence(baseline, rows, p["validation_seeds"])
    m.s.write(directory / "validation.json", result)
    independent, _ = review.verify_dataset(directory / "validation", p["validation_seeds"])
    review.assert_comparison(review.independent_comparison(baseline, independent, p["validation_seeds"]), result)
    print(f"Verified validation: total={total}, seed={seed}, checks={result['checks']}", flush=True)
    return directory, result


def all_pass(results):
    assert set(results) == {"0", "1", "2"}
    return all(result["passed"] for result in results.values())


def final_test(parents, total, p):
    assert m.s.read(OUT / "selection.json")["total_steps"] == total
    hashes = {str(seed): m.s.digest(path / "policy.zip") for seed, path in parents.items()}
    m.s.write(OUT / "tuning_closed.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
             "total_steps": total, "checkpoint_sha256": hashes, "final_seeds": p["final_seeds"]})
    baseline = m.evaluate(None, OUT / "final/pd", p["final_seeds"], {})
    results = {}
    for seed, parent in parents.items():
        assert m.s.digest(parent / "policy.zip") == hashes[str(seed)]
        agent = PPO.load(parent / "policy.zip", device="cpu")
        rows = m.evaluate(agent, OUT / f"final/ppo_seed_{seed}", p["final_seeds"], p["environment"])
        results[str(seed)] = m.evidence(baseline, rows, p["final_seeds"])
    m.s.write(OUT / "final/acceptance.json", {"passed": all_pass(results), "total_steps": total, "training_seeds": results})


def audit():
    p = verify()
    design = m.s.read(OUT / "design.json")
    for path, sha in design["prior_artifacts_sha256"].items():
        assert m.s.digest(m.s.ROOT / path) == sha, path
    test_log = (OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "Ran 32 tests" in test_log and "\nOK" in test_log and "FAILED" not in test_log
    baseline, _ = review.verify_dataset(m.OUT / "validation/pd", p["validation_seeds"])
    history = {262144: {str(seed): m.s.read(path / "validation.json") for seed, path in initial_parents().items()}}
    raw_counts, rows = {}, []
    for total in p["total_step_checkpoints"]:
        folder = OUT / f"round_{total}"
        if not folder.exists():
            continue
        results = {}
        for seed in p["training_seeds"]:
            directory = folder / f"seed_{seed}"
            episodes, count = review.verify_dataset(directory / "validation", p["validation_seeds"])
            raw_counts[f"round_{total}/seed_{seed}"] = count
            independent = review.independent_comparison(baseline, episodes, p["validation_seeds"])
            recorded = m.s.read(directory / "validation.json")
            review.assert_comparison(independent, recorded)
            meta, config = m.s.read(directory / "training.json"), m.s.read(directory / "config.json")
            assert meta["actual_total_steps"] == total and meta["optimizer_steps_after"] > meta["optimizer_steps_before"]
            assert meta["additional_steps"] == total - config["previous_total_steps"]
            assert meta["checkpoint_sha256"] == m.s.digest(directory / "policy.zip")
            parent = initial_parents()[seed] if total == 524288 else OUT / f"round_524288/seed_{seed}"
            assert meta["parent_checkpoint_sha256"] == m.s.digest(parent / "policy.zip")
            results[str(seed)] = recorded
        summary = m.s.read(folder / "summary.json")
        assert summary["training_seeds"] == results and summary["passed"] == all_pass(results)
        history[total] = results
    passing = [total for total in p["total_step_checkpoints"] if total in history and all_pass(history[total])]
    selected = passing[0] if passing else None
    outcome = m.s.read(OUT / "outcome.json")
    assert outcome["selected_total_steps"] == selected
    final_results = None
    if selected is None:
        assert list(history) == [262144, 524288, 1048576]
        assert not (OUT / "selection.json").exists() and not (OUT / "final").exists()
        assert not outcome["final_test_run"]
    else:
        assert m.s.read(OUT / "selection.json")["total_steps"] == selected
        if selected == 524288:
            assert not (OUT / "round_1048576").exists()
        close = m.s.read(OUT / "tuning_closed.json")
        final_pd, count = review.verify_dataset(OUT / "final/pd", p["final_seeds"])
        raw_counts["final/pd"] = count
        final_results = m.s.read(OUT / "final/acceptance.json")
        for seed in p["training_seeds"]:
            episodes, count = review.verify_dataset(OUT / f"final/ppo_seed_{seed}", p["final_seeds"])
            raw_counts[f"final/ppo_seed_{seed}"] = count
            independent = review.independent_comparison(final_pd, episodes, p["final_seeds"])
            review.assert_comparison(independent, final_results["training_seeds"][str(seed)])
            assert close["checkpoint_sha256"][str(seed)] == m.s.digest(OUT / f"round_{selected}/seed_{seed}/policy.zip")
        assert final_results["passed"] == all_pass(final_results["training_seeds"])
    for total, results in history.items():
        for seed, result in results.items():
            for group in result["groups"]:
                rows.append(f"| {total:,} | {seed} | {group['condition']} | {group['pd_mean_deg']:.4f} | {group['ppo_mean_deg']:.4f} | {group['reduction_pct']:.2f}% | {5 - group['ppo_falls']}/5 | {'N/A' if group['condition'] == 'healthy' else str(sum(q['passed'] for q in group['peak_pairs'])) + '/5'} |")
    font = Path(r"C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    budgets = list(history)
    for i, (ax, title) in enumerate(zip(axes.flat, ["Front left", "Front right", "Rear left", "Rear right"], strict=True)):
        for seed, color, marker in [("0", "#355c7d", "o"), ("1", "#62815c", "s"), ("2", "#9a6544", "^")]:
            values = [history[total][seed]["groups"][i + 1]["reduction_pct"] for total in budgets]
            ax.plot(np.array(budgets) / 1000, values, marker=marker, color=color, label=f"Model {int(seed) + 1}")
        ax.axhline(40, color="black", linestyle="--", linewidth=1, label="Required: 40%")
        ax.axhline(0, color="#999999", linewidth=.5)
        ax.set_title(title)
        ax.set_xticks(np.array(budgets) / 1000, [f"{n/1000:.0f}" for n in budgets])
        ax.set_xlabel("Total training steps (thousands)")
        ax.set_ylabel("Reduction in mean body tilt (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Does more training improve balance across knee motors?", fontsize=20)
    fig.text(.5, .015, "Old validation conditions only. Each model has a separate training seed. Means do not establish peak or survival acceptance.", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.savefig(OUT / "validation_progress.png", dpi=160)
    plt.close(fig)
    final_status = "not run: no common validation success" if final_results is None else ("PASS" if final_results["passed"] else "FAIL")
    final_rows = []
    if final_results is not None:
        for seed, result in final_results["training_seeds"].items():
            for group in result["groups"]:
                final_rows.append(f"| {seed} | {group['condition']} | {group['pd_mean_deg']:.4f} | {group['ppo_mean_deg']:.4f} | {group['reduction_pct']:.2f}% | {20 - group['ppo_falls']}/20 | {'N/A' if group['condition'] == 'healthy' else str(sum(q['passed'] for q in group['peak_pairs'])) + '/20'} |")
    final_section = ("## Held-out comparison\n\n| Training seed | Condition | PD mean tilt (°) | Learned mean tilt (°) | Reduction | Survived 30 s | Peak checks |\n"
                     "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(final_rows)) if final_rows else "## Held-out comparison\n\nNot run: no common validation success within the fixed budget. Seeds 700–719 remain unused."
    report = f"""# Continued multi-motor training

**Final evaluation: {final_status}.** Configuration B was continued without changing its controller, reward, architecture or PPO hyperparameters. The earliest permitted common validation success was {f'{selected:,} steps' if selected else 'not reached within the fixed budget'}. All failures remain in the data.

![Validation progress](validation_progress.png)

| Total steps per model | Training seed | Validation condition | PD mean tilt (°) | Learned mean tilt (°) | Reduction vs PD | Survived 30 s | Paired peak checks |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

These comparisons use only the old validation seeds 100–104 and their unchanged paired PD reference. Healthy means cover (0,30] seconds; fault means cover (5,30] seconds. Every checkpoint must separately survive every 30-second episode, keep healthy tilt at most PD +0.3°, reduce each faulted motor's mean by at least 40%, and have no larger peak in any paired fault episode. No mean can compensate for a peak failure or a failed training seed. Model labels 1–3 correspond to training seeds 0–2. The plot shows means, not full acceptance.

The starting budget was 262,144 steps. Checkpoints were examined only at total budgets 524,288 and 1,048,576 per seed. All three models were continued at each required stage. Common success would stop later training and allow exactly one held-out comparison using new seeds 700–719. If no stage passes, those seeds remain unused and no final directory is created. Previous final seeds 600–619 are retired and do not control selection. No new configuration was searched.

Continuation restores the saved weights, Adam optimizer and timestep count. Global and environment random generators restart with the original training seed at each stage because their earlier streams were not saved. This is a documented continuation procedure, not bit-exact uninterrupted training. `training.json` records optimizer counters before/after, changed parameters, additional steps, actual wall time, peak process memory, checkpoint hash and parent hash. Checkpoint reload predictions were verified identical. Training remains local, without CUDA or simulation rendering.

The new protocol was saved before training at {p['created_utc']}. It preserves the original training distribution: healthy operation or one of four knee faults, randomized strength and onset, without privileged fault identity in the observation. PD gains, physical model, joint limits, residual gate, fault evaluation schedule, and measurement definitions remain unchanged. All {len(design['prior_artifacts_sha256'])} protected earlier artifacts are unchanged. The earlier failed report and checkpoints remain available under `results/multimotor`.

{final_section}

The 32 functional/regression tests passed before continuation. Each new validation was independently checked from the full raw physics trace before proceeding. The final audit recomputed means with compensated sums, peaks per episode, recovery windows, pairing, budget selection and checkpoint lineage using the existing independent review functions. Exact results and raw sample counts are in `data_validation.json`; `round_*/summary.json` preserves every validation check. Protocol, configurations, source hashes, monitor logs, weights and training metadata are retained. {('Held-out details are in `final/acceptance.json` and `final/*/trace.csv`.' if final_results is not None else 'Reserved final seeds 700–719 were not used. This is a failed validation experiment, not a held-out acceptance claim.')}

To recheck the persisted evidence, run `.venv\\Scripts\\python.exe multimotor_extend.py audit`. Preparation and execution refuse overwriting this experiment. The evidence remains limited to a flat simulated floor, small initial-position variations, and one knee motor weakened at a time. More training within these budgets cannot establish that any further budget would succeed, nor isolate reward or controller design as the sole cause of a failure. Walking, simultaneous faults, other joint types and hardware transfer remain untested. No new video was made.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    proof = {"completed": True, "selected_total_steps": selected, "final_test_run": final_results is not None,
             "final_acceptance_passed": None if final_results is None else final_results["passed"],
             "reserved_final_seeds_unused": final_results is None, "raw_dataset_rows": raw_counts,
             "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]), "tests_passed_before_training": 32,
             "independent_calculations_verified": True, "optimizer_continuation_verified": True,
             "training_budget_and_selection_verified": True, "cuda": False, "rendering": False}
    target = OUT / "data_validation.json"
    if target.exists():
        assert m.s.read(target) == proof
    else:
        m.s.write(target, proof)
    print("Audit complete:", final_status, flush=True)


def run():
    p = verify()
    m.s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
             "torch": torch.__version__, "cuda_build": torch.version.cuda, "device": "cpu", "rendering": False})
    parents, previous, selected = initial_parents(), p["initial_total_steps"], None
    for total in p["total_step_checkpoints"]:
        next_parents, results = {}, {}
        for seed in p["training_seeds"]:
            directory, result = continue_seed(parents[seed], previous, total, seed, p)
            next_parents[seed], results[str(seed)] = directory, result
        summary = {"total_steps": total, "passed": all_pass(results), "training_seeds": results,
                   "selection_uses_final_seeds": False}
        m.s.write(OUT / f"round_{total}/summary.json", summary)
        parents, previous = next_parents, total
        print(f"Common validation at {total}: {summary['passed']}", flush=True)
        if summary["passed"]:
            selected = total
            m.s.write(OUT / "selection.json", summary)
            final_test(parents, total, p)
            break
    m.s.write(OUT / "outcome.json", {"selected_total_steps": selected, "final_test_run": selected is not None,
             "max_budget_validation_failed": selected is None})
    audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "audit"])
    {"prepare": prepare, "run": run, "audit": audit}[parser.parse_args().stage]()
