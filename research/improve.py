"""Four frozen PPO configurations, validation-only selection, and held-out audit."""
from project_paths import project_file

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from go2_env import Go2Balance
from run import evaluate, require_baseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/improvement"
PROTOCOL = OUT / "protocol.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(baseline, learned, seeds, protocol):
    expected = {(seed, strength) for seed in seeds for strength in protocol["strengths"]}
    for rows in (baseline, learned):
        if len(rows) != len(expected) or {(r["seed"], r["final_strength"]) for r in rows} != expected:
            raise ValueError("Evaluation grid is incomplete or duplicated")
    reference = {(r["seed"], r["final_strength"]): r for r in baseline}
    for row in learned:
        if row["initial_state_sha256"] != reference[row["seed"], row["final_strength"]]["initial_state_sha256"]:
            raise ValueError("Initial states do not match")
    criteria = protocol["criteria"]
    survival = all(not r["fallen"] and abs(r["simulated_s"] - criteria["episode_seconds"]) < 1e-6
                   for r in baseline + learned)
    groups = []
    for strength in protocol["strengths"]:
        before = [r for r in baseline if r["final_strength"] == strength]
        after = [r for r in learned if r["final_strength"] == strength]
        groups.append({"strength": strength,
                       "baseline_mean_tilt_deg": float(np.mean([r["mean_tilt_deg"] for r in before])),
                       "ppo_mean_tilt_deg": float(np.mean([r["mean_tilt_deg"] for r in after])),
                       "baseline_falls": sum(r["fallen"] for r in before),
                       "ppo_falls": sum(r["fallen"] for r in after)})
    healthy, severe = groups[0], groups[-1]
    healthy_delta = healthy["ppo_mean_tilt_deg"] - healthy["baseline_mean_tilt_deg"]
    reduction = 1 - severe["ppo_mean_tilt_deg"] / severe["baseline_mean_tilt_deg"]
    checks = {"survival": survival,
              "healthy": healthy["ppo_mean_tilt_deg"] <= healthy["baseline_mean_tilt_deg"] + criteria["healthy_max_extra_tilt_deg"],
              "severe": severe["ppo_mean_tilt_deg"] <= (1 - criteria["severe_min_reduction"]) * severe["baseline_mean_tilt_deg"]}
    return {"passed": all(checks.values()), "checks": checks, "groups": groups,
            "healthy_delta_deg": healthy_delta, "severe_reduction": reduction,
            "paired_initial_states_verified": True}


def train(config_id, seed, directory, protocol, design):
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {directory}")
    directory.mkdir(parents=True)
    parameters = protocol["ppo"]
    config = protocol["configurations"][config_id]
    write(directory / "config.json", {"configuration_id": config_id, "training_seed": seed,
          "environment": config, "ppo": parameters, "design": design,
          "python": platform.python_version(), "sb3": stable_baselines3.__version__,
          "torch": torch.__version__, "cuda_build": torch.version.cuda})
    env = Go2Balance(training=True, **config)
    agent = PPO("MlpPolicy", Monitor(env, str(directory / "monitor.csv")), device="cpu", seed=seed,
                n_steps=parameters["n_steps"], batch_size=parameters["batch_size"],
                n_epochs=parameters["n_epochs"], learning_rate=parameters["learning_rate"],
                policy_kwargs={"net_arch": parameters["net_arch"],
                               "log_std_init": parameters["log_std_init"]}, verbose=1)
    start = time.perf_counter()
    agent.learn(total_timesteps=parameters["steps"])
    if not all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters()):
        raise RuntimeError("Non-finite trained policy")
    agent.save(directory / "policy")
    observation, _ = env.reset(seed=seed)
    restored = PPO.load(directory / "policy", device="cpu")
    np.testing.assert_array_equal(agent.predict(observation, deterministic=True)[0],
                                  restored.predict(observation, deterministic=True)[0])
    memory = psutil.Process().memory_info()
    write(directory / "training.json", {"steps": agent.num_timesteps,
          "wall_s": time.perf_counter() - start,
          "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
          "checkpoint_sha256": digest(directory / "policy.zip"), "reload_predictions_identical": True})
    env.close()
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "tune", "select", "replicate", "final"])
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    torch.set_num_threads(protocol["ppo"]["torch_threads"])
    if torch.version.cuda is not None:
        raise RuntimeError("This experiment requires a CPU-only PyTorch build")
    assert set(protocol["validation_seeds"]).isdisjoint(protocol["final_seeds"])
    require_baseline()
    design = {"protocol_sha256": digest(PROTOCOL),
              "source_sha256": {name: digest(project_file(ROOT, name)) for name in ("go2_env.py", "run.py", "improve.py")},
              "requirements_sha256": digest(ROOT / "requirements-lock.txt"),
              "model_revision": subprocess.check_output(
                  ["git", "-C", str(ROOT / "vendor/mujoco_menagerie"), "rev-parse", "HEAD"], text=True).strip()}
    if args.stage == "prepare":
        if (OUT / "design.json").exists():
            raise FileExistsError("Experiment already frozen")
        write(OUT / "design.json", design)
        rows = evaluate(None, OUT / "validation_pd_trace.csv", seeds=protocol["validation_seeds"], trace_every=1)
        check = evidence(rows, rows, protocol["validation_seeds"], protocol)
        if not check["checks"]["survival"]:
            raise RuntimeError("PD survival gate failed")
        write(OUT / "validation_pd.json", rows)
        print("Protocol and source frozen; validation PD complete", flush=True)
        return
    if read(OUT / "design.json") != design:
        raise RuntimeError("Frozen protocol, source, dependency lock, or model revision changed")
    if args.stage == "tune":
        if args.value not in protocol["configurations"]:
            parser.error("tune requires A, B, C, or D")
        if (OUT / "selection.json").exists():
            raise RuntimeError("Selection is frozen; no further tuning")
        directory = OUT / "tuning" / args.value
        policy = train(args.value, 0, directory, protocol, design)
        rows = evaluate(policy, directory / "validation_trace.csv", seeds=protocol["validation_seeds"],
                        env_kwargs=protocol["configurations"][args.value], trace_every=1)
        result = evidence(read(OUT / "validation_pd.json"), rows, protocol["validation_seeds"], protocol)
        write(directory / "validation.json", {"episodes": rows, **result})
        print(json.dumps(result), flush=True)
    elif args.stage == "select":
        if (OUT / "selection.json").exists():
            raise FileExistsError("Selection already frozen")
        candidates = {key: read(OUT / "tuning" / key / "validation.json") for key in protocol["configurations"]}
        passing = [key for key in candidates if candidates[key]["passed"]]
        winner = min(passing, key=lambda key: (candidates[key]["groups"][-1]["ppo_mean_tilt_deg"],
                     candidates[key]["groups"][0]["ppo_mean_tilt_deg"], key)) if passing else None
        write(OUT / "selection.json", {"selected": winner, "validation_results": candidates,
              "design": design, "selection_uses_final_seeds": False})
        print("Selected:", winner, flush=True)
    elif args.stage == "replicate":
        winner = read(OUT / "selection.json")["selected"]
        if winner is None or args.value not in ("1", "2"):
            raise ValueError("A passing selection and training seed 1 or 2 are required")
        train(winner, int(args.value), OUT / "replicates" / f"seed_{args.value}", protocol, design)
    else:
        winner = read(OUT / "selection.json")["selected"]
        if winner is None:
            raise RuntimeError("No validation candidate passed; final seeds remain unused")
        directories = {0: OUT / "tuning" / winner,
                       1: OUT / "replicates/seed_1", 2: OUT / "replicates/seed_2"}
        for directory in directories.values():
            metadata = read(directory / "training.json")
            if metadata["checkpoint_sha256"] != digest(directory / "policy.zip"):
                raise RuntimeError("Checkpoint changed")
        final = OUT / "final"
        final.mkdir()  # Never silently rerun or overwrite the held-out evaluation.
        write(final / "evaluation_started.json", {"selected": winner, "design": design,
              "seeds": protocol["final_seeds"], "tuning_closed": True})
        baseline = evaluate(None, final / "pd_trace.csv", seeds=protocol["final_seeds"], trace_every=1)
        write(final / "pd.json", baseline)
        results = {}
        for seed, directory in directories.items():
            policy = PPO.load(directory / "policy", device="cpu")
            rows = evaluate(policy, final / f"ppo_seed_{seed}_trace.csv", seeds=protocol["final_seeds"],
                            env_kwargs=protocol["configurations"][winner], trace_every=1)
            result = evidence(baseline, rows, protocol["final_seeds"], protocol)
            write(final / f"ppo_seed_{seed}.json", {"episodes": rows, **result})
            results[str(seed)] = result
        write(final / "acceptance.json", {"passed": all(r["passed"] for r in results.values()),
              "selected": winner, "training_seeds": results, "design": design})
        print(json.dumps(read(final / "acceptance.json")), flush=True)


if __name__ == "__main__":
    main()
