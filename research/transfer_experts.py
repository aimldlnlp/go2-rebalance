"""One preregistered DAgger transfer from fixed experts to unprivileged actors."""
from project_paths import project_file

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import time

import numpy as np
import torch
from stable_baselines3 import PPO

import expert_selector as expert
import multimotor as m
import multimotor_extend as extension
import review_multimotor as review

OUT = m.s.ROOT / "results/expert_transfer"
ROUNDS = 4
EPISODES = 10
EPOCHS = 20
BATCH = 256


def rollout_seeds(seed, round_id):
    return list(range(10000 + seed * 1000 + round_id * EPISODES,
                      10000 + seed * 1000 + (round_id + 1) * EPISODES))


def gates(observations):
    return np.clip(np.rad2deg(np.arccos(np.clip(-observations[:, 2], -1., 1.))) - 2., 0., 1.).astype(np.float32)


def prepare():
    assert not OUT.exists(), "Refusing to overwrite an experiment"
    p = expert.verify()
    assert m.s.read(expert.OUT / "completion_audit.json")["diagnostic_passed"]
    for name, sha in m.s.read(expert.OUT / "evidence_manifest.json").items():
        assert m.s.digest(expert.OUT / name) == sha, name
    used, protected = m.s.inventory()
    used.update(m.s.read(m.s.OUT / "protocol.json")["evaluation_seeds"])
    for path in m.s.OUT.rglob("*"):
        if path.is_file():
            protected[str(path.relative_to(m.s.ROOT))] = m.s.digest(path)
    training_starts = [s for seed in range(3) for r in range(ROUNDS) for s in rollout_seeds(seed, r)]
    assert len(training_starts) == len(set(training_starts)) == 120
    assert not set(training_starts).intersection(used)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "DAgger with four cumulative supervised fits. Round 0 uses deterministic teacher behavior; rounds 1-3 use only the current deterministic student. Query the fixed teacher on every visited pre-action observation, including student mistakes. No teacher action blending and no PPO reward optimization.",
        "student": {"architecture": [64, 64], "observations": 60, "actions": 12,
            "initialization": "Independent random actor initialization with seeds 0,1,2; no expert weight copying. Same expert library, separate trajectories and optimizers.",
            "container": "SB3 PPO MlpPolicy ZIP for existing predict/evaluate compatibility. Only actor means are trained by supervised loss; value network/log_std are unchanged. num_timesteps remains 0 because PPO.learn is never called."},
        "supervision": "12 clipped deterministic expert actions; healthy teacher is zero. Mean-squared error on actor mean, weighted by the square of the unchanged tilt gate computed from the ordinary gravity observation. This compares applied residuals before joint/torque clipping and avoids contradictory irrelevant actions when gate is zero.",
        "rounds": ROUNDS, "episodes_per_round": EPISODES, "epochs_per_round": EPOCHS, "batch_size": BATCH,
        "optimizer": {"name": "Adam", "learning_rate": .0003, "gradient_clip_norm": .5},
        "training_seeds": [0, 1, 2], "training_reset_seeds": training_starts,
        "training_distribution": "Two 30-second episodes per condition per round, new reset seeds per student/round/episode. Fixed healthy plus separate knee faults, 100% to 10% at 5s, flat floor and original reset noise. This transfer targets the tested diagnostic schedule, not randomized strengths or onset generalization. Truncate fallen episodes, retain their samples and report them.",
        "mapping": p["mapping"], "checkpoints": p["checkpoints"], "environment": p["environment"],
        "criteria": p["criteria"], "measurement": p["measurement"],
        "validation_seeds": list(range(100, 105)), "reserved_final_seeds": list(range(700, 720)),
        "retired_diagnostic_seeds": p["diagnostic_seeds"], "prior_recorded_seeds": sorted(used),
        "selection": "Evaluate only round 3 for all three students on old validation 100-104. All original criteria must pass separately for each motor and each training seed. No intermediate checkpoint selection, retraining, remapping or extra configurations.",
        "final_gate": "Only if all three final-round students pass validation, freeze checkpoint hashes and run PD plus all three students once on 700-719. Otherwise keep final seeds unused and create no final directory. Do not tune after final results.",
        "execution": {"device": "cpu", "torch_threads": 1, "cuda": False, "rendering": False},
        "failure_policy": "Preserve every round, dataset, optimizer, loss log, rollout summary and failed result. Fixed four-round budget for all three students even if one validation fails."}
    sources = ["transfer_experts.py", "test_transfer_experts.py", "expert_selector.py", "multimotor.py",
               "multimotor_extend.py", "review_multimotor.py", "go2_env.py", "sudden.py", "requirements-lock.txt"]
    source_hashes = {name: m.s.digest(project_file(m.s.ROOT, name)) for name in sources}
    model_hashes = m.s.read(expert.OUT / "design.json")["model_sha256"]
    OUT.mkdir()
    m.s.write(OUT / "protocol.json", protocol)
    m.s.write(OUT / "design.json", {"protocol_sha256": m.s.digest(OUT / "protocol.json"),
        "source_sha256": source_hashes, "model_sha256": model_hashes, "prior_artifacts_sha256": protected})
    print(f"Frozen: three independent students, four rounds x ten episodes, twenty epochs; {len(protected)} prior artifacts", flush=True)


def verify():
    design = m.s.read(OUT / "design.json")
    assert m.s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for name, sha in {**design["source_sha256"], **design["model_sha256"]}.items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    p = m.s.read(OUT / "protocol.json")
    assert p["rounds"] == ROUNDS and p["episodes_per_round"] == EPISODES
    assert p["epochs_per_round"] == EPOCHS and p["batch_size"] == BATCH
    assert p["environment"] == expert.verify()["environment"]
    assert p["criteria"] == m.verify()["criteria"]
    assert p["mapping"] == expert.MAPPING
    assert not set(p["training_reset_seeds"]).intersection(p["prior_recorded_seeds"] + p["reserved_final_seeds"])
    assert set(p["reserved_final_seeds"]) == set(range(700, 720))
    for checkpoint in p["checkpoints"].values():
        assert m.s.digest(m.s.ROOT / checkpoint["path"]) == checkpoint["sha256"]
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def collect(student, teachers, seed, round_id, config):
    observations, labels, conditions, reset_ids, control_ticks, rows = [], [], [], [], [], []
    for episode, start in enumerate(rollout_seeds(seed, round_id)):
        condition = m.CONDITIONS[episode % len(m.CONDITIONS)]
        teacher = teachers[condition]
        env = m.MultiMotorBalance(motor=condition, **config)
        try:
            observation, _ = env.reset(seed=start)
            for tick in range(env.horizon):
                target = np.zeros(12, dtype=np.float32) if teacher is None else teacher.predict(observation, deterministic=True)[0]
                action = target if round_id == 0 else student.predict(observation, deterministic=True)[0]
                observations.append(observation.copy())
                labels.append(target.copy())
                conditions.append(m.CONDITIONS.index(condition))
                reset_ids.append(start)
                control_ticks.append(tick)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            rows.append({"condition": condition, "seed": start, "control_steps": tick + 1,
                         "simulated_s": info["time_s"], "fallen": info["fallen"],
                         "behavior": "teacher" if round_id == 0 else "student"})
        finally:
            env.close()
        print(f"collect seed={seed} round={round_id} {condition}/{start}: steps={tick+1} fallen={info['fallen']}", flush=True)
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    assert x.shape == (len(y), 60) and y.shape[1:] == (12,)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.abs(y).max() <= 1
    return {"observations": x, "targets": y, "gates": gates(x),
            "condition_ids": np.asarray(conditions, dtype=np.int8),
            "reset_seeds": np.asarray(reset_ids), "control_ticks": np.asarray(control_ticks)}, rows


def fit(student, data, epochs, batch_size, rng):
    x, y, gate = [torch.as_tensor(data[k]) for k in ("observations", "targets", "gates")]
    student.policy.set_training_mode(True)
    losses = []
    for _ in range(epochs):
        total, weight_sum = 0., 0.
        for indices in np.array_split(rng.permutation(len(x)), max(1, (len(x) + batch_size - 1) // batch_size)):
            weight = gate[indices, None].square()
            prediction = student.policy.get_distribution(x[indices]).distribution.mean
            weighted_error = ((prediction - y[indices]).square() * weight).sum()
            denominator = (12 * weight.sum()).clamp_min(1.)
            loss = weighted_error / denominator
            assert torch.isfinite(loss)
            student.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.policy.parameters(), .5, error_if_nonfinite=True)
            student.policy.optimizer.step()
            total += float(weighted_error.detach())
            weight_sum += float(12 * weight.sum())
        losses.append(total / max(weight_sum, 1.))
    student.policy.set_training_mode(False)
    assert all(torch.isfinite(parameter).all() for parameter in student.policy.parameters())
    return losses


def train(seed, teachers, p):
    directory = OUT / f"seed_{seed}"
    directory.mkdir()
    env = m.MultiMotorBalance(**p["environment"])
    student = PPO("MlpPolicy", env, device="cpu", seed=seed, learning_rate=.0003,
                  policy_kwargs={"net_arch": [64, 64], "log_std_init": -2.}, verbose=0)
    unchanged = {name: parameter.detach().clone() for name, parameter in student.policy.named_parameters()
                 if name.startswith(("mlp_extractor.value_net", "value_net", "log_std"))}
    m.s.write(directory / "config.json", {"training_seed": seed, "environment": p["environment"],
        "protocol_sha256": m.s.digest(OUT / "protocol.json"), "observations": 60, "fault_identity_input": False,
        "supervised_actor_only": True, "initialized_from_expert_weights": False})
    rng = np.random.default_rng(seed)
    aggregate, total_steps = None, 0
    started = time.perf_counter()
    for r in range(ROUNDS):
        folder = directory / f"round_{r}"
        folder.mkdir()
        data, episodes = collect(student, teachers, seed, r, p["environment"])
        with (folder / "dataset.npz").open("xb") as handle:
            np.savez_compressed(handle, **data)
        m.s.write(folder / "rollouts.json", episodes)
        aggregate = data if aggregate is None else {k: np.concatenate((aggregate[k], data[k])) for k in data}
        losses = fit(student, aggregate, EPOCHS, BATCH, rng)
        total_steps += len(data["observations"])
        student.save(folder / "policy.zip")
        restored = PPO.load(folder / "policy.zip", device="cpu")
        sample = data["observations"][:100]
        np.testing.assert_array_equal(student.predict(sample, deterministic=True)[0], restored.predict(sample, deterministic=True)[0])
        for name, before in unchanged.items():
            assert torch.equal(before, dict(student.policy.named_parameters())[name])
        m.s.write(folder / "training.json", {"round": r, "new_samples": len(data["observations"]),
            "cumulative_samples": len(aggregate["observations"]), "total_environment_control_steps": total_steps,
            "epoch_applied_action_mse": losses, "dataset_sha256": m.s.digest(folder / "dataset.npz"),
            "checkpoint_sha256": m.s.digest(folder / "policy.zip"), "reload_predictions_identical": True,
            "critic_and_log_std_unchanged": True, "ppo_learn_called": False})
        print(f"FIT seed={seed} round={r}: cumulative={len(aggregate['observations'])}, mse={losses[-1]:.6g}", flush=True)
    env.close()
    m.s.write(directory / "training.json", {"wall_s": time.perf_counter() - started,
        "environment_control_steps": total_steps, "rounds_completed": ROUNDS,
        "checkpoint_sha256": m.s.digest(directory / "round_3/policy.zip")})
    return student


def evaluated(policy, folder, seeds, config, baseline=None):
    rows = m.evaluate(policy, folder, seeds, config)
    verified, count = review.verify_dataset(folder, seeds)
    if baseline is None:
        return rows, count
    result = m.evidence(baseline, rows, seeds)
    review.assert_comparison(review.independent_comparison(baseline, verified, seeds), result)
    m.s.write(folder / "acceptance.json", result)
    return result, count


def final_test(results, p):
    assert extension.all_pass(results), "Final seeds require all three validation passes"
    assert not (OUT / "final").exists(), "Refusing to repeat final evaluation"
    checkpoints = {str(seed): m.s.digest(OUT / f"seed_{seed}/round_3/policy.zip") for seed in range(3)}
    m.s.write(OUT / "tuning_closed.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoints, "validation_passed": True})
    baseline, _ = evaluated(None, OUT / "final/pd", p["reserved_final_seeds"], p["environment"])
    final = {}
    for seed in range(3):
        path = OUT / f"seed_{seed}/round_3/policy.zip"
        assert m.s.digest(path) == checkpoints[str(seed)]
        student = PPO.load(path, device="cpu")
        final[str(seed)], _ = evaluated(student, OUT / f"final/seed_{seed}", p["reserved_final_seeds"], p["environment"], baseline)
    result = {"passed": extension.all_pass(final), "training_seeds": final}
    m.s.write(OUT / "final/acceptance.json", result)
    return result


def run():
    assert not (OUT / "execution.json").exists(), "Refusing to rerun training"
    p = verify()
    tests = (OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "\nOK" in tests and "FAILED" not in tests
    teachers_by_seed = {int(seed): PPO.load(m.s.ROOT / item["path"], device="cpu")
                        for seed, item in p["checkpoints"].items()}
    teachers = {condition: None if seed is None else teachers_by_seed[seed] for condition, seed in p["mapping"].items()}
    m.s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "torch_threads": torch.get_num_threads(), "cuda_build": torch.version.cuda,
        "rendering": False, "tests_sha256": m.s.digest(OUT / "tests-before.log"),
        "versions": {name: version(name) for name in ["torch", "mujoco", "stable-baselines3", "numpy"]}})
    baseline, _ = evaluated(None, OUT / "validation/pd", p["validation_seeds"], p["environment"])
    results = {}
    for seed in p["training_seeds"]:
        student = train(seed, teachers, p)
        results[str(seed)], _ = evaluated(student, OUT / f"validation/seed_{seed}", p["validation_seeds"], p["environment"], baseline)
        print(f"VALIDATION seed={seed}: {results[str(seed)]['checks']}", flush=True)
    summary = {"passed": extension.all_pass(results), "training_seeds": results}
    m.s.write(OUT / "validation/acceptance.json", summary)
    final = final_test(results, p) if summary["passed"] else None
    m.s.write(OUT / "outcome.json", {"validation_passed": summary["passed"], "final_test_run": final is not None,
        "final_acceptance_passed": None if final is None else final["passed"],
        "finished_utc": datetime.now(timezone.utc).isoformat()})
    audit()


def audit():
    p = verify()
    design = m.s.read(OUT / "design.json")
    execution = m.s.read(OUT / "execution.json")
    assert datetime.fromisoformat(p["created_utc"]) < datetime.fromisoformat(execution["started_utc"])
    assert execution["tests_sha256"] == m.s.digest(OUT / "tests-before.log")
    teachers = {int(seed): PPO.load(m.s.ROOT / item["path"], device="cpu") for seed, item in p["checkpoints"].items()}
    counts, training_counts = {}, {}
    for seed in range(3):
        cumulative = 0
        for r in range(ROUNDS):
            folder = OUT / f"seed_{seed}/round_{r}"
            meta = m.s.read(folder / "training.json")
            assert meta["dataset_sha256"] == m.s.digest(folder / "dataset.npz")
            assert meta["checkpoint_sha256"] == m.s.digest(folder / "policy.zip")
            with np.load(folder / "dataset.npz", allow_pickle=False) as data:
                x, y = data["observations"], data["targets"]
                assert x.shape == (len(y), 60) and y.shape == (len(x), 12)
                assert np.isfinite(x).all() and np.isfinite(y).all() and np.abs(y).max() <= 1
                np.testing.assert_array_equal(gates(x), data["gates"])
                assert set(np.unique(data["reset_seeds"])) == set(rollout_seeds(seed, r))
                episodes = m.s.read(folder / "rollouts.json")
                assert len(episodes) == EPISODES
                for episode, row in enumerate(episodes):
                    condition = m.CONDITIONS[episode % 5]
                    assert row["condition"] == condition and row["seed"] == rollout_seeds(seed, r)[episode]
                    mask = data["reset_seeds"] == row["seed"]
                    assert mask.sum() == row["control_steps"]
                    np.testing.assert_array_equal(data["control_ticks"][mask], np.arange(row["control_steps"]))
                    assert np.all(data["condition_ids"][mask] == episode % 5)
                    assert row["behavior"] == ("teacher" if r == 0 else "student")
                for cid, condition in enumerate(m.CONDITIONS):
                    mask = data["condition_ids"] == cid
                    selected = p["mapping"][condition]
                    expected = np.zeros_like(y[mask]) if selected is None else teachers[selected].predict(x[mask], deterministic=True)[0]
                    np.testing.assert_allclose(expected, y[mask], rtol=0, atol=2e-5)
                cumulative += len(x)
                assert meta["new_samples"] == len(x) and meta["cumulative_samples"] == cumulative
                assert len(meta["epoch_applied_action_mse"]) == EPOCHS
                assert np.isfinite(meta["epoch_applied_action_mse"]).all()
                training_counts[f"seed_{seed}/round_{r}"] = len(x)
        assert m.s.read(OUT / f"seed_{seed}/training.json")["environment_control_steps"] == cumulative
    validation = m.s.read(OUT / "validation/acceptance.json")
    baseline, counts["validation/pd"] = review.verify_dataset(OUT / "validation/pd", p["validation_seeds"])
    for seed in range(3):
        folder = OUT / f"validation/seed_{seed}"
        rows, counts[f"validation/seed_{seed}"] = review.verify_dataset(folder, p["validation_seeds"])
        review.assert_comparison(review.independent_comparison(baseline, rows, p["validation_seeds"]), validation["training_seeds"][str(seed)])
        assert m.s.read(folder / "acceptance.json") == validation["training_seeds"][str(seed)]
    assert validation["passed"] == extension.all_pass(validation["training_seeds"])
    outcome = m.s.read(OUT / "outcome.json")
    assert outcome["validation_passed"] == validation["passed"]
    final = None
    if validation["passed"]:
        closed = m.s.read(OUT / "tuning_closed.json")
        final = m.s.read(OUT / "final/acceptance.json")
        baseline, counts["final/pd"] = review.verify_dataset(OUT / "final/pd", p["reserved_final_seeds"])
        for seed in range(3):
            assert closed["checkpoint_sha256"][str(seed)] == m.s.digest(OUT / f"seed_{seed}/round_3/policy.zip")
            folder = OUT / f"final/seed_{seed}"
            rows, counts[f"final/seed_{seed}"] = review.verify_dataset(folder, p["reserved_final_seeds"])
            review.assert_comparison(review.independent_comparison(baseline, rows, p["reserved_final_seeds"]), final["training_seeds"][str(seed)])
        assert final["passed"] == extension.all_pass(final["training_seeds"])
        assert outcome["final_test_run"] and outcome["final_acceptance_passed"] == final["passed"]
    else:
        assert not (OUT / "final").exists() and not (OUT / "tuning_closed.json").exists()
        assert not outcome["final_test_run"] and outcome["final_acceptance_passed"] is None
    for name, sha in design["prior_artifacts_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    proof = {"completed": True, "validation_passed": validation["passed"], "final_test_run": final is not None,
        "final_acceptance_passed": None if final is None else final["passed"],
        "reserved_final_seeds_unused": final is None, "training_samples": training_counts,
        "raw_dataset_rows": counts, "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]),
        "independent_raw_metrics_verified": True, "teacher_labels_verified": True,
        "student_observations": 60, "student_fault_identity_input": False, "cuda": False, "rendering": False}
    table = []
    for seed, result in validation["training_seeds"].items():
        for group in result["groups"]:
            c = group["condition"]
            table.append(f"| {seed} | {c} | {group['pd_mean_deg']:.4f} | {group['ppo_mean_deg']:.4f} | {group['reduction_pct']:.2f}% | {5-group['ppo_falls']}/5 | {sum(q['passed'] for q in group['peak_pairs'])}/5 |")
    report = f"""# Expert-to-student transfer

**Validation: {'PASS' if validation['passed'] else 'FAIL'}. Final evaluation: {'not run; seeds 700-719 unused' if final is None else ('PASS' if final['passed'] else 'FAIL')}. Three independently initialized actors were trained using one frozen DAgger configuration, without motor identity in their inputs.

| Student training seed | Condition | PD mean (degrees) | Student mean (degrees) | Reduction | No fall | Paired peaks |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Method and fixed scope

Four rounds per student, ten episodes per round (two per condition), twenty Adam epochs per cumulative dataset, batch 256, learning rate 0.0003, gradient clipping 0.5. First round uses teacher behavior; all later rounds use deterministic student behavior with expert labels on the visited observations. No PPO reward training, expert parameter initialization, online teacher selection at inference, or new dependency. A fresh 64x64 SB3 actor takes the original 60 observations and emits 12 residuals. Critic/log_std stay unchanged; the ZIP is a compatible inference container, not a PPO-trained checkpoint. `num_timesteps=0` is intentional; actual supervised samples and environment steps are recorded separately.

Expert mapping is FL/RL -> seed 2, FR -> seed 0, RR -> seed 1, healthy -> zero. Only the label generator uses scenario identity; neither student inputs nor its runtime controller receive it. Loss compares actor means with clipped expert actions, weighted by the square of the existing tilt gate derived from the ordinary gravity observation. Gates, joint/torque limits, PD gains, reward and physical model are unchanged; residual scale remains 0.35 rad. Training targets only the fixed 10%-at-5s fault schedule; strength/onset generalization was not studied.

Training initialization seeds are 0,1,2, with 120 distinct fresh reset seeds recorded before execution. Only the final (round 3) checkpoint was validated on old seeds 100-104; there was no selection among rounds or tuning after validation. Every seed must separately survive all episodes, stay within healthy PD +0.3 degrees, reduce every motor mean by at least 40%, and satisfy each fault paired peak <= PD. Means use (0,30] for healthy and (5,30] for faults; peaks cover full episodes. No pooling across models or motors. Final seeds 700-719 are conditional on all-three validation success; diagnostic seeds 800-804 were not used for training or selection.

## Evidence

Protocol frozen at {p['created_utc']}; execution started at {execution['started_utc']}. `design.json` protects source/model/checkpoint lineage and {proof['prior_artifacts_unchanged']} prior artifacts, all unchanged. `tests-before.log` and its execution hash preserve pre-run test evidence. Per-student/round NPZ datasets contain observations, expert targets, gate weights, condition/reset/tick provenance; `rollouts.json` keeps falls and durations. Dataset condition IDs are audit metadata only. Training logs retain all losses, optimizer checkpoints, hashes and exact save/reload prediction checks. Expert target labels were rechecked with batched inference (absolute tolerance 2e-5 for batch-versus-single float32 arithmetic).

Independent raw-trace review covered {sum(counts.values()):,} validation/final physics samples: condition/seed grids, contiguous ticks, timing, fault strength boundaries, finite values, fall flags, means, peaks, recovery, paired initial qpos hashes and unchanged acceptance. `completion_audit.json`, `validation/acceptance.json`, individual acceptance files and any permitted `final/acceptance.json` retain exact outcomes. Loss values describe imitation on visited states, not closed-loop acceptance or generalization.

## Interpretation

{'All three students passed this fixed validation; the one-shot held-out result is reported separately in final/acceptance.json. Do not retune from those final results.' if validation['passed'] else 'Transfer did not satisfy the all-three validation gate. Preserve the failures and do not claim reliable single-policy compensation; final seeds remain unused. This bounded result does not isolate architecture, observation sufficiency or optimizer behavior as the cause.'}

The same expert library is shared across the independently trained students. Five repeatedly used validation starts, flat floor, small reset noise and fixed fault strength/onset limit inference. A successful privileged selector does not guarantee that its behavior is learnable or recoverable from every student-visited state. No hardware, walking, simultaneous faults, video, or rendering was tested.

Recheck: `.venv\\Scripts\\python.exe transfer_experts.py audit`. Preparation and training refuse overwriting existing work.
"""
    for path, value in [(OUT / "completion_audit.json", proof)]:
        if path.exists():
            assert m.s.read(path) == value
        else:
            m.s.write(path, value)
    target = OUT / "REPORT.md"
    if target.exists():
        assert target.read_text(encoding="utf-8") == report
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(report)
    print("\n".join(table), flush=True)
    print(f"AUDIT complete: validation={validation['passed']}, final={None if final is None else final['passed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "audit"])
    {"prepare": prepare, "run": run, "audit": audit}[parser.parse_args().stage]()
