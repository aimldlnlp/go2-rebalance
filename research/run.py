"""Stage gates, CPU benchmark, training, and paired held-out evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil

from go2_env import Go2Balance

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = [100, 101, 102, 103, 104]
STRENGTHS = [1.0, 0.75, 0.5, 0.25, 0.1]


def save(name, value):
    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def episode(env, seed, policy=None, trace=None, trace_every=10):
    observation, info = env.reset(seed=seed)
    initial_hash = hashlib.sha256(env.data.qpos.tobytes()).hexdigest()
    tilts = []
    total_reward = 0.0
    started = time.perf_counter()
    process = psutil.Process()
    rss = process.memory_info().rss
    for step in range(env.horizon):
        action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        tilts.append(info["tilt_deg"])
        if step % 50 == 0:
            rss = max(rss, process.memory_info().rss)
        if trace is not None and (step % trace_every == 0 or terminated or truncated):
            trace.writerow({"seed": seed, "final_strength": env.strength, **info})
        if terminated or truncated:
            break
    wall = time.perf_counter() - started
    memory = process.memory_info()
    return {"seed": seed, "final_strength": env.strength, "initial_state_sha256": initial_hash,
            "simulated_s": info["time_s"], "fallen": info["fallen"],
            "mean_tilt_deg": float(np.mean(tilts)), "max_tilt_deg": float(np.max(tilts)),
            "final_xy_drift_m": float(np.linalg.norm(env.data.qpos[:2])),
            "return": total_reward, "wall_s": wall, "realtime_factor": info["time_s"] / wall,
            "control_steps_per_s": (step + 1) / wall,
            "sampled_rss_mib": rss / 2**20,
            "process_peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20}


def evaluate(policy, filename, seeds=SEEDS, env_kwargs=None, trace_every=10):
    path = RESULTS / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    env = Go2Balance(**(env_kwargs or {}))
    rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "final_strength", "time_s", "tilt_deg",
                                                     "height_m", "fallen", "strength"])
        writer.writeheader()
        for strength in STRENGTHS:
            env.strength = strength
            for seed in seeds:
                row = episode(env, seed, policy, writer, trace_every)
                rows.append(row)
            group = rows[-len(seeds):]
            print(f"strength={strength:g}: falls={sum(r['fallen'] for r in group)}/{len(group)}, "
                  f"mean tilt={np.mean([r['mean_tilt_deg'] for r in group]):.3f} deg", flush=True)
    env.close()
    return rows


def require_baseline():
    path = RESULTS / "baseline.json"
    if not path.exists() or not json.loads(path.read_text())["passed"]:
        raise RuntimeError("Run and pass the baseline stage first")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "stress", "train", "compare"])
    parser.add_argument("--steps", type=int, default=65536)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.stage == "baseline":
        import mujoco
        env = Go2Balance()
        rows = [episode(env, seed) for seed in SEEDS]
        passed = all(not r["fallen"] and r["simulated_s"] >= 30 - 1e-6 for r in rows)
        model_revision = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1] / "vendor/mujoco_menagerie"), "rev-parse", "HEAD"],
            text=True).strip()
        result = {"passed": passed, "python": platform.python_version(), "mujoco": mujoco.__version__,
                  "model_revision": model_revision, "pd_kp": 50.0, "pd_kd": 3.0,
                  "physics_dt_s": env.model.opt.timestep, "control_dt_s": env.dt,
                  "rendering": False, "episodes": rows}
        save("baseline.json", result)
        print(json.dumps(result, indent=2))
        if not passed:
            raise RuntimeError("Baseline failed: do not proceed to motor faults or training")
    elif args.stage == "stress":
        require_baseline()
        rows = evaluate(None, "baseline_trace.csv")
        save("stress.json", {"motor": "FL_calf", "onset_s": 5, "ramp_s": 10, "episodes": rows})
    elif args.stage == "train":
        require_baseline()
        if not (RESULTS / "stress.json").exists():
            raise RuntimeError("Run the stress stage first")
        import torch
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        torch.set_num_threads(1)
        env = Go2Balance(training=True)
        check_env(env)
        agent = PPO("MlpPolicy", Monitor(env, str(RESULTS / "training_monitor.csv")),
                    device="cpu", seed=0, n_steps=1024, batch_size=64, n_epochs=5,
                    learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]}, verbose=1)
        started = time.perf_counter()
        agent.learn(total_timesteps=args.steps)
        if not all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters()):
            raise RuntimeError("Training produced non-finite policy parameters")
        agent.save(RESULTS / "ppo_go2")
        save("training.json", {"requested_steps": args.steps, "actual_steps": agent.num_timesteps,
                               "wall_s": time.perf_counter() - started, "training_seed": 0,
                               "device": str(agent.device), "torch": torch.__version__,
                               "stable_baselines3": stable_baselines3.__version__,
                               "torch_cuda_build": torch.version.cuda,
                               "peak_rss_mib": getattr(psutil.Process().memory_info(), "peak_wset",
                                                       psutil.Process().memory_info().rss) / 2**20})
        env.close()
    else:
        require_baseline()
        import torch
        import stable_baselines3
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        policy = PPO.load(RESULTS / "ppo_go2", device="cpu")
        if not all(torch.isfinite(parameter).all() for parameter in policy.policy.parameters()):
            raise RuntimeError("Checkpoint contains non-finite parameters")
        baseline = json.loads((RESULTS / "stress.json").read_text())["episodes"]
        learned = evaluate(policy, "ppo_trace.csv")
        for before, after in zip(baseline, learned, strict=True):
            assert (before["seed"], before["final_strength"], before["initial_state_sha256"]) == (
                after["seed"], after["final_strength"], after["initial_state_sha256"])
        groups = []
        for strength in STRENGTHS:
            before = [r for r in baseline if r["final_strength"] == strength]
            after = [r for r in learned if r["final_strength"] == strength]
            groups.append({"strength": strength, "episodes_per_controller": len(before),
                           "baseline_falls": sum(r["fallen"] for r in before),
                           "ppo_falls": sum(r["fallen"] for r in after),
                           "baseline_mean_tilt_deg": float(np.mean([r["mean_tilt_deg"] for r in before])),
                           "ppo_mean_tilt_deg": float(np.mean([r["mean_tilt_deg"] for r in after])),
                           "baseline_mean_survival_s": float(np.mean([r["simulated_s"] for r in before])),
                           "ppo_mean_survival_s": float(np.mean([r["simulated_s"] for r in after]))})
        save("comparison.json", {"paired_initial_states_verified": True,
                                 "stable_baselines3": stable_baselines3.__version__, "groups": groups,
                                 "ppo_episodes": learned})
        print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
