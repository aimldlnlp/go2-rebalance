"""Four preregistered multi-motor PPO candidates and a three-seed held-out test."""
from project_paths import project_file

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from go2_env import Go2Balance
import sudden as s

OUT = s.ROOT / "results/multimotor"
MOTORS = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
CONDITIONS = ["healthy", *MOTORS]


class MultiMotorBalance(Go2Balance):
    def __init__(self, motor="healthy", randomize=False, record=False, tilt_weight=5.0, **kwargs):
        if motor not in CONDITIONS or not np.isfinite(tilt_weight) or tilt_weight < 5:
            raise ValueError("Invalid motor or tilt weight")
        self.motor, self.randomize, self.record = motor, randomize, record
        self.onset_tick, self.tilt_weight, self.samples = 2500, tilt_weight, []
        super().__init__(strength=1.0 if motor == "healthy" else .1, **kwargs)
        self.weak_motor = self.model.actuator("FL_calf" if motor == "healthy" else motor).id

    def motor_strength(self):
        return 1.0 if round(self.data.time / s.DT) < self.onset_tick else self.strength

    def reset(self, **kwargs):
        self.samples = []
        observation, _ = super().reset(**kwargs)
        if self.randomize:
            self.motor = str(self.np_random.choice(CONDITIONS))
            self.strength = 1.0 if self.motor == "healthy" else float(
                self.np_random.choice([.1, .25, .5, .75], p=[.5, .2, .2, .1]))
            self.onset_tick = round(float(self.np_random.uniform(3, 8)) / s.DT)
            self.weak_motor = self.model.actuator("FL_calf" if self.motor == "healthy" else self.motor).id
        return observation, self.metrics()

    def metrics(self):
        info = super().metrics()
        tick = round(self.data.time / s.DT)
        if self.record and (not self.samples or tick > self.samples[-1][0]):
            self.samples.append((tick, tick * s.DT, info["tilt_deg"], info["height_m"], info["fallen"],
                                 info["strength"], 1.0 if tick <= self.onset_tick else self.strength))
        return {**info, "fault_motor": self.motor, "final_strength": self.strength,
                "fault_onset_s": self.onset_tick * s.DT}

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward -= (self.tilt_weight - 5) * np.deg2rad(info["tilt_deg"])**2
        return observation, float(reward), terminated, truncated, info


def prepare():
    assert not OUT.exists(), "Refusing to overwrite an experiment"
    used, files = s.inventory()
    used.update(s.read(s.OUT / "protocol.json")["evaluation_seeds"])
    for path in s.OUT.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(s.ROOT))] = s.digest(path)
    final_seeds = list(range(600, 620))
    assert not used.intersection(final_seeds)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": CONDITIONS, "validation_seeds": list(range(100, 105)),
        "final_seeds": final_seeds, "prior_used_seeds": sorted(used), "training_seeds": [0, 1, 2],
        "configurations": {
            "A": {"residual_scale": .25, "residual_penalty": .03, "tilt_deadband_deg": 2.0, "tilt_weight": 5.0},
            "B": {"residual_scale": .35, "residual_penalty": .03, "tilt_deadband_deg": 2.0, "tilt_weight": 5.0},
            "C": {"residual_scale": .25, "residual_penalty": .03, "tilt_deadband_deg": 2.0, "tilt_weight": 15.0},
            "D": {"residual_scale": .35, "residual_penalty": .03, "tilt_deadband_deg": 2.0, "tilt_weight": 15.0}},
        "ppo": {"steps": 262144, "n_steps": 1024, "batch_size": 64, "n_epochs": 5,
                "learning_rate": .0003, "net_arch": [64, 64], "log_std_init": -2.0},
        "pilot": {"configuration": "A", "training_seed": 0, "steps": 16384, "evaluation_seeds": [100],
                  "gate": "Same candidate, not an extra configuration. Save and reload after 16384 steps, verify finite parameters and identical predictions, run all five conditions on validation seed 100, then continue the same agent to 262144 total steps. Pilot failures against performance criteria are recorded, not a reason to change thresholds. Implementation or nonfinite-state errors stop the run."},
        "training_distribution": "From scratch; at each reset choose uniformly among healthy and four knee motors. For faults, choose remaining strength 0.1/0.25/0.5/0.75 with probabilities 0.5/0.2/0.2/0.1, and onset uniformly from 3 to 8 seconds quantized to physics ticks. Step change, one motor per episode. Initial noise and physical model unchanged. No fault identity/strength/onset added to observations; no action permutation or mirrored policy. No pretrained weights used.",
        "evaluation": "Healthy plus four separate knee-motor faults. Every fault changes 100% to 10% at exactly 5 seconds. 30-second episodes, deterministic policy, paired initial hashes, 500 Hz measurement and 50 Hz policy. PD evaluated once per seed/condition and shared across checkpoints.",
        "criteria": {"duration_s": 30, "healthy_max_extra_deg": .3, "minimum_fault_mean_reduction": .4,
                     "maximum_paired_fault_peak_ratio": 1.0},
        "measurement": "Healthy mean: all physics samples in (0,30]. Fault mean: samples in (5,30], then equal-weight mean across evaluation seeds independently per motor and training seed. Peak: full observed episode including initial sample. A shortened episode always fails survival. Recovery uses unchanged sudden.py definition and is diagnostic only. No pooling between faulted motors or trained checkpoints, no tolerance added to tilt thresholds.",
        "selection": "Train exactly A-D with training seed 0 at the same fixed 262144-step budget. Use validation seeds 100-104 only. Lexicographic rank: number of failed checks (survival, healthy, and mean/peak independently for each motor); worst motor PPO/PD mean ratio; total failed peak pairs; healthy mean excess; configuration ID. Freeze the lowest rank. Even if all fail, retain the best candidate as a failed validation selection and test its three training seeds for an honest result, without claiming acceptance or changing criteria.",
        "replication": "Train only the selected configuration from scratch at 262144 steps with seeds 1 and 2. No further configurations, budget extension, or tuning. After all three checkpoints exist, close tuning, freeze their hashes, and evaluate once on fresh seeds 600-619. Report all failures; final criteria must pass independently for all three checkpoints and every motor.",
        "execution": {"device": "cpu", "torch_threads": 1, "cuda": False, "rendering": False},
        "evidence": "Preserve prior artifacts and weights by hash. Keep protocol, source/requirement hashes, configuration, monitor logs, checkpoints, raw traces, all candidate validations, held-out comparisons, independent Data-method validation and an English scientific comparison figure. Reuse the project's Markdown report format. No video or dashboard."}
    OUT.mkdir(parents=True)
    s.write(OUT / "protocol.json", protocol)
    sources = ["multimotor.py", "test_multimotor.py", "go2_env.py", "sudden.py", "requirements-lock.txt"]
    s.write(OUT / "design.json", {"protocol_sha256": s.digest(OUT / "protocol.json"),
             "source_sha256": {name: s.digest(project_file(s.ROOT, name)) for name in sources},
             "prior_artifacts_sha256": files})
    print("Frozen: four configurations, 262144 steps each, validation 100-104, final 600-619", flush=True)


def verify():
    p = s.read(OUT / "protocol.json")
    design = s.read(OUT / "design.json")
    assert s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for path, sha in design["source_sha256"].items():
        assert s.digest(s.ROOT / path) == sha, path
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def evaluate(policy, folder, seeds, config):
    folder.mkdir(parents=True)
    rows = []
    with (folder / "trace.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", *s.FIELDS])
        for condition in CONDITIONS:
            env = MultiMotorBalance(motor=condition, record=True, **config)
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                initial = s.digest_bytes(env.data.qpos.tobytes())
                for _ in range(env.horizon):
                    action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                assert env.samples[-1][0] > s.ONSET
                writer.writerows((condition, seed, *sample) for sample in env.samples)
                handle.flush()
                row = {"condition": condition, "seed": seed, "initial_state_sha256": initial,
                       **s.summarize(env.samples),
                       "mean_tilt_deg": float(np.mean([r[2] for r in env.samples[1:]]))}
                rows.append(row)
            env.close()
            group = rows[-len(seeds):]
            print(f"{folder.name}/{condition}: mean_post={np.mean([r['mean_post_tilt_deg'] for r in group]):.3f} falls={sum(r['fallen'] for r in group)}/{len(seeds)}", flush=True)
    s.write(folder / "episodes.json", rows)
    return rows


def evidence(baseline, learned, seeds):
    expected = {(c, seed) for c in CONDITIONS for seed in seeds}
    for rows in (baseline, learned):
        assert len(rows) == len(expected) and {(r["condition"], r["seed"]) for r in rows} == expected
    reference = {(r["condition"], r["seed"]): r for r in baseline}
    for r in learned:
        assert r["initial_state_sha256"] == reference[r["condition"], r["seed"]]["initial_state_sha256"]
    checks = {"survival": all(not r["fallen"] and r["simulated_s"] == 30 for r in baseline + learned)}
    groups = []
    for condition in CONDITIONS:
        before = [r for r in baseline if r["condition"] == condition]
        after = [r for r in learned if r["condition"] == condition]
        field = "mean_tilt_deg" if condition == "healthy" else "mean_post_tilt_deg"
        b, a = float(np.mean([r[field] for r in before])), float(np.mean([r[field] for r in after]))
        peak_pairs = [{"seed": b0["seed"], "pd_peak_deg": b0["max_tilt_deg"], "ppo_peak_deg": a0["max_tilt_deg"],
                       "passed": a0["max_tilt_deg"] <= b0["max_tilt_deg"]} for b0, a0 in zip(before, after, strict=True)]
        if condition == "healthy":
            checks["healthy"] = a <= b + .3
        else:
            checks[condition + "_mean"] = a <= .6 * b
            checks[condition + "_peak"] = all(r["passed"] for r in peak_pairs)
        groups.append({"condition": condition, "pd_mean_deg": b, "ppo_mean_deg": a,
                       "reduction_pct": 100 * (1 - a / b), "delta_deg": a - b,
                       "ppo_falls": sum(r["fallen"] for r in after), "peak_pairs": peak_pairs})
    return {"passed": all(checks.values()), "checks": checks, "groups": groups,
            "paired_initial_states_verified": True}


def train(config_id, seed, directory, protocol):
    directory.mkdir(parents=True)
    config, parameters = protocol["configurations"][config_id], protocol["ppo"]
    s.write(directory / "config.json", {"configuration_id": config_id, "training_seed": seed,
            "environment": config, "ppo": parameters, "device": "cpu", "cuda_build": torch.version.cuda,
            "protocol_sha256": s.digest(OUT / "protocol.json"), "from_scratch": True})
    env = MultiMotorBalance(randomize=True, **config)
    monitor = Monitor(env, str(directory / "monitor.csv"), info_keywords=("fault_motor", "final_strength", "fault_onset_s"))
    kwargs = {key: parameters[key] for key in ["n_steps", "batch_size", "n_epochs", "learning_rate"]}
    agent = PPO("MlpPolicy", monitor, device="cpu", seed=seed, **kwargs,
                policy_kwargs={"net_arch": parameters["net_arch"], "log_std_init": parameters["log_std_init"]}, verbose=0)
    started = time.perf_counter()
    if config_id == "A" and seed == 0:
        agent.learn(total_timesteps=16384)
        agent.save(directory / "pilot_policy")
        restored = PPO.load(directory / "pilot_policy", device="cpu")
        check = MultiMotorBalance(**config)
        observation, _ = check.reset(seed=100)
        np.testing.assert_array_equal(agent.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
        check.close()
        assert all(torch.isfinite(x).all() for x in agent.policy.parameters())
        evaluate(restored, directory / "pilot", [100], config)
        print("Pilot passed finite-state and checkpoint checks; continuing A", flush=True)
        agent.learn(total_timesteps=parameters["steps"] - 16384, reset_num_timesteps=False)
    else:
        agent.learn(total_timesteps=parameters["steps"])
    assert agent.num_timesteps == parameters["steps"]
    assert all(torch.isfinite(x).all() for x in agent.policy.parameters())
    agent.save(directory / "policy")
    restored = PPO.load(directory / "policy", device="cpu")
    check = MultiMotorBalance(**config)
    observation, _ = check.reset(seed=100)
    np.testing.assert_array_equal(agent.predict(observation, deterministic=True)[0], restored.predict(observation, deterministic=True)[0])
    check.close()
    monitor.close()
    memory = psutil.Process().memory_info()
    s.write(directory / "training.json", {"actual_steps": agent.num_timesteps,
            "wall_s_including_pilot_checks": time.perf_counter() - started,
            "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
            "checkpoint_sha256": s.digest(directory / "policy.zip"), "reload_predictions_identical": True})
    print(f"Trained {config_id}/seed_{seed}: {agent.num_timesteps} steps", flush=True)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    if parser.parse_args().stage == "prepare":
        prepare()
        return
    p = verify()
    s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cpu", "torch": torch.__version__, "cuda_build": torch.version.cuda, "rendering": False})
    pd_val = evaluate(None, OUT / "validation/pd", p["validation_seeds"], {})
    candidates = {}
    for config_id in p["configurations"]:
        directory = OUT / "candidates" / config_id
        agent = train(config_id, 0, directory, p)
        rows = evaluate(agent, directory / "validation", p["validation_seeds"], p["configurations"][config_id])
        result = evidence(pd_val, rows, p["validation_seeds"])
        s.write(directory / "validation.json", result)
        candidates[config_id] = result
        print(f"Validation {config_id}: {result['checks']}", flush=True)
    def rank(key):
        r = candidates[key]
        return (sum(not v for v in r["checks"].values()),
                max(g["ppo_mean_deg"] / g["pd_mean_deg"] for g in r["groups"][1:]),
                sum(not pair["passed"] for g in r["groups"][1:] for pair in g["peak_pairs"]),
                r["groups"][0]["delta_deg"], key)
    selected = min(candidates, key=rank)
    s.write(OUT / "selection.json", {"selected": selected, "validation_passed": candidates[selected]["passed"],
            "ranking": {key: rank(key) for key in candidates}, "validation_results": candidates,
            "selection_uses_final_seeds": False})
    print(f"Selected {selected}; validation passed={candidates[selected]['passed']}", flush=True)
    directories = {0: OUT / "candidates" / selected}
    for seed in [1, 2]:
        directory = OUT / "replicates" / f"seed_{seed}"
        agent = train(selected, seed, directory, p)
        rows = evaluate(agent, directory / "validation", p["validation_seeds"], p["configurations"][selected])
        s.write(directory / "validation.json", evidence(pd_val, rows, p["validation_seeds"]))
        directories[seed] = directory
    s.write(OUT / "tuning_closed.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "selected": selected, "checkpoint_sha256": {str(seed): s.digest(path / "policy.zip") for seed, path in directories.items()},
            "final_seeds": p["final_seeds"]})
    baseline = evaluate(None, OUT / "final/pd", p["final_seeds"], {})
    results = {}
    for seed, directory in directories.items():
        agent = PPO.load(directory / "policy", device="cpu")
        rows = evaluate(agent, OUT / f"final/ppo_seed_{seed}", p["final_seeds"], p["configurations"][selected])
        results[str(seed)] = evidence(baseline, rows, p["final_seeds"])
    s.write(OUT / "final/acceptance.json", {"passed": all(r["passed"] for r in results.values()),
            "selected": selected, "training_seeds": results})
    print("Final evaluation finished:", {seed: result["passed"] for seed, result in results.items()}, flush=True)


if __name__ == "__main__":
    main()
