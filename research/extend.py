"""Continue frozen configuration A; validate consistency before a fresh held-out test."""
from project_paths import project_file

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from go2_env import Go2Balance
from improve import OUT, ROOT, PROTOCOL, digest, evidence, read, write
from run import evaluate

EXT = OUT / "extended"
PLAN = OUT / "extension_protocol.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "select", "final"])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--total", type=int, choices=[262144, 393216, 524288])
    args = parser.parse_args()
    protocol, plan = read(PROTOCOL), read(PLAN)
    torch.set_num_threads(1)
    if torch.version.cuda is not None:
        raise RuntimeError("CPU-only PyTorch required")
    assert set(plan["final_seeds"]).isdisjoint(plan["retired_test_seeds"] + plan["validation_seeds"])
    original = read(OUT / "design.json")
    for name, checksum in original["source_sha256"].items():
        if digest(project_file(ROOT, name)) != checksum:
            raise RuntimeError("Original frozen implementation changed")
    if digest(PROTOCOL) != original["protocol_sha256"]:
        raise RuntimeError("Original criteria or configuration changed")
    design = {"parent_design_sha256": digest(OUT / "design.json"),
              "plan_sha256": digest(PLAN), "runner_sha256": digest(Path(__file__)),
              "requirements_sha256": digest(ROOT / "requirements-lock.txt")}
    config = protocol["configurations"][plan["configuration_id"]]
    if args.stage == "prepare":
        EXT.mkdir(exist_ok=True)
        if (EXT / "design.json").exists():
            raise FileExistsError("Extension already frozen")
        write(EXT / "design.json", design)
        return
    if read(EXT / "design.json") != design:
        raise RuntimeError("Frozen extension changed")
    if args.stage == "train":
        if args.seed is None or args.total is None:
            parser.error("train requires --seed and --total")
        if (EXT / "selection.json").exists():
            raise RuntimeError("Training is closed after common validation success")
        previous = args.total - plan["additional_steps_per_round"]
        if previous == protocol["ppo"]["steps"]:
            parent = OUT / "tuning/A" if args.seed == 0 else OUT / "replicates" / f"seed_{args.seed}"
        else:
            if not (EXT / f"round_{previous}/summary.json").exists():
                raise RuntimeError("Complete the preceding common validation round first")
            parent = EXT / f"round_{previous}" / f"seed_{args.seed}"
        directory = EXT / f"round_{args.total}" / f"seed_{args.seed}"
        directory.mkdir(parents=True)
        write(directory / "config.json", {"training_seed": args.seed, "environment": config,
              "ppo": protocol["ppo"], "target_total_steps": args.total, "design": design,
              "parent_checkpoint_sha256": digest(parent / "policy.zip"), "cuda_build": torch.version.cuda,
              "rng_restart_at_round": True})
        env = Go2Balance(training=True, **config)
        agent = PPO.load(parent / "policy", env=Monitor(env, str(directory / "monitor.csv")), device="cpu")
        if agent.num_timesteps != previous or agent.seed != args.seed:
            raise RuntimeError("Incorrect parent checkpoint or training seed")
        agent.set_random_seed(args.seed)
        start = time.perf_counter()
        agent.learn(total_timesteps=plan["additional_steps_per_round"], reset_num_timesteps=False)
        assert agent.num_timesteps == args.total
        if not all(torch.isfinite(parameter).all() for parameter in agent.policy.parameters()):
            raise RuntimeError("Non-finite policy")
        agent.save(directory / "policy")
        restored = PPO.load(directory / "policy", device="cpu")
        observation, _ = env.reset(seed=args.seed)
        np.testing.assert_array_equal(agent.predict(observation, deterministic=True)[0],
                                      restored.predict(observation, deterministic=True)[0])
        memory = psutil.Process().memory_info()
        write(directory / "training.json", {"steps": agent.num_timesteps,
              "wall_s": time.perf_counter() - start,
              "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
              "checkpoint_sha256": digest(directory / "policy.zip"), "reload_predictions_identical": True})
        rows = evaluate(restored, directory / "validation_trace.csv", seeds=plan["validation_seeds"],
                        env_kwargs=config, trace_every=1)
        result = evidence(read(OUT / "validation_pd.json"), rows, plan["validation_seeds"], protocol)
        write(directory / "validation.json", {"episodes": rows, **result})
        print(json.dumps({"seed": args.seed, "total": args.total, **result}), flush=True)
    elif args.stage == "select":
        if args.total is None:
            parser.error("select requires --total")
        if (EXT / "selection.json").exists():
            raise FileExistsError("Common checkpoint already selected")
        results = {str(seed): read(EXT / f"round_{args.total}" / f"seed_{seed}/validation.json")
                   for seed in plan["training_seeds"]}
        result = {"total_steps": args.total, "passed": all(r["passed"] for r in results.values()),
                  "training_seeds": results, "design": design, "selection_uses_final_seeds": False}
        write(EXT / f"round_{args.total}/summary.json", result)
        if result["passed"]:
            write(EXT / "selection.json", result)
        print(json.dumps({"total": args.total, "passed": result["passed"]}), flush=True)
    else:
        selection = read(EXT / "selection.json")
        total = selection["total_steps"]
        final = EXT / "final"
        final.mkdir()
        write(final / "evaluation_started.json", {"total_steps": total, "seeds": plan["final_seeds"],
              "design": design, "tuning_closed": True})
        baseline = evaluate(None, final / "pd_trace.csv", seeds=plan["final_seeds"], trace_every=1)
        write(final / "pd.json", baseline)
        results = {}
        for seed in plan["training_seeds"]:
            directory = EXT / f"round_{total}" / f"seed_{seed}"
            if read(directory / "training.json")["checkpoint_sha256"] != digest(directory / "policy.zip"):
                raise RuntimeError("Checkpoint changed")
            policy = PPO.load(directory / "policy", device="cpu")
            rows = evaluate(policy, final / f"ppo_seed_{seed}_trace.csv", seeds=plan["final_seeds"],
                            env_kwargs=config, trace_every=1)
            result = evidence(baseline, rows, plan["final_seeds"], protocol)
            write(final / f"ppo_seed_{seed}.json", {"episodes": rows, **result})
            results[str(seed)] = result
        write(final / "acceptance.json", {"passed": all(r["passed"] for r in results.values()),
              "total_steps": total, "training_seeds": results, "design": design})
        print(json.dumps(read(final / "acceptance.json")), flush=True)


if __name__ == "__main__":
    main()
