"""Frozen, inference-only sudden motor-loss experiment. Existing experiments stay intact."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from go2_env import Go2Balance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sudden"
DT = .002
ONSET = 2500
FIELDS = ["tick", "time_s", "tilt_deg", "height_m", "fallen", "strength_next_step", "strength_applied"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


class SuddenBalance(Go2Balance):
    def __init__(self, **kwargs):
        super().__init__(strength=.1, **kwargs)
        assert self.model.opt.timestep == DT and self.horizon == 1500
        self.samples = []

    def motor_strength(self):
        # Integer physics ticks avoid a late switch from accumulated floating-point time.
        return 1.0 if round(self.data.time / DT) < ONSET else .1

    def reset(self, **kwargs):
        self.samples = []
        return super().reset(**kwargs)

    def metrics(self):
        info = super().metrics()
        tick = round(self.data.time / DT)
        if not self.samples or tick > self.samples[-1][0]:
            self.samples.append((tick, tick * DT, info["tilt_deg"], info["height_m"],
                                 info["fallen"], info["strength"], 1.0 if tick <= ONSET else .1))
        return info


def summarize(samples):
    a = np.asarray(samples, dtype=float)
    ticks, tilts = a[:, 0], a[:, 2]
    post = tilts[ticks > ONSET]
    reference = float(np.mean(tilts[(ticks >= 2000) & (ticks <= ONSET)]))
    bound = reference + .5
    post_ticks = ticks[ticks > ONSET]
    recovery = None
    start = None
    for tick, tilt in zip(post_ticks, post, strict=True):
        if tilt > bound:
            start = None
        elif start is None:
            start = tick
        if start is not None and tick - start >= 500:
            recovery = float((start - ONSET) * DT)
            break
    if len(post) and np.all(post <= bound):
        recovery = 0.0
    return {"mean_post_tilt_deg": float(np.mean(post)), "max_tilt_deg": float(np.max(tilts)),
            "max_post_tilt_deg": float(np.max(post)), "pre_fault_mean_tilt_deg": reference,
            "recovery_bound_deg": bound, "recovery_s": recovery,
            "recovered": recovery is not None, "fallen": bool(np.any(a[:, 4])),
            "simulated_s": float(a[-1, 1]), "physics_samples": len(samples)}


def criteria(pd, learned):
    assert len(pd) == len(learned) == 20
    pairs = []
    for before, after in zip(pd, learned, strict=True):
        assert before["seed"] == after["seed"]
        assert before["initial_state_sha256"] == after["initial_state_sha256"]
        pairs.append({"seed": before["seed"],
                      "pd_max_tilt_deg": before["max_tilt_deg"],
                      "ppo_max_tilt_deg": after["max_tilt_deg"],
                      "peak_passed": after["max_tilt_deg"] <= before["max_tilt_deg"],
                      "mean_reduction_pct": 100 * (1 - after["mean_post_tilt_deg"] / before["mean_post_tilt_deg"])})
    before = float(np.mean([r["mean_post_tilt_deg"] for r in pd]))
    after = float(np.mean([r["mean_post_tilt_deg"] for r in learned]))
    checks = {"survival": all(not r["fallen"] and r["simulated_s"] == 30 for r in pd + learned),
              "post_fault_mean": after <= .6 * before,
              "every_paired_peak": all(r["peak_passed"] for r in pairs)}
    return {"passed": all(checks.values()), "checks": checks,
            "pd_mean_post_tilt_deg": before, "ppo_mean_post_tilt_deg": after,
            "reduction_pct": 100 * (1 - after / before), "pairs": pairs,
            "recovered_episodes": sum(r["recovered"] for r in learned),
            "recovery_times_s": [r["recovery_s"] for r in learned]}


def inventory():
    used = set()
    files = {}
    def collect(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)
        elif isinstance(value, list):
            for v in value:
                collect(v, key)
        elif "seed" in key and isinstance(value, int) and not isinstance(value, bool):
            used.add(value)
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or OUT in path.parents:
            continue
        files[str(path.relative_to(ROOT))] = digest(path)
        if path.suffix == ".json":
            collect(read(path))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(line for line in handle if not line.startswith("#")):
                    if "seed" in row:
                        used.add(int(row["seed"]))
    return used, files


def prepare():
    assert not OUT.exists(), "Refusing to overwrite an existing experiment"
    used, files = inventory()
    seeds = list(range(400, 420))
    assert not used.intersection(seeds)
    assert read(ROOT / "results/improvement/extended/selection.json")["total_steps"] == 262144
    checkpoints = {}
    for seed in range(3):
        path = ROOT / f"results/improvement/extended/round_262144/seed_{seed}/policy.zip"
        config = read(path.parent / "config.json")
        assert config["environment"] == {"residual_scale": .25, "residual_penalty": .03, "tilt_deadband_deg": 2.0}
        checkpoints[f"ppo_seed_{seed}"] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                                          "training_seed": seed, "environment": config["environment"]}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_seeds": seeds,
        "previously_used_seeds": sorted(used), "video_seed": 400,
        "video_controllers": ["pd", *checkpoints], "checkpoints": checkpoints,
        "fault": {"motor": "FL_calf", "onset_s": 5, "before": 1.0, "after": .1,
                  "schedule": "100% on physics intervals [0,5); 10% on [5,30]. No ramp."},
        "episode_s": 30, "physics_dt_s": DT, "control_dt_s": .02,
        "measurement": "500 Hz using the unchanged environment metrics after each physics step. Integer tick times remove roundoff. Post-fault samples are (5,30], exactly 12500 per completed episode. Mean is per episode, then equally across 20 seeds, independently for each checkpoint. Peak criterion uses all [0,30] samples, including the healthy phase. Post-fault peaks are also saved. No numeric slack on tilt criteria.",
        "criteria": {"all_episodes_30s_no_fall": True, "minimum_mean_reduction": .4,
                     "maximum_paired_peak_ratio": 1.0},
        "recovery": {"reference": "Mean tilt at physics ticks 2000..2500 inclusive, [4,5] seconds, separately for each controller/seed.",
                     "extra_deg": .5, "hold_s": 1.0,
                     "definition": "First start time strictly after 5s of a continuous interval of at least 1s with tilt <= reference + 0.5 degrees; elapsed from 5s. If never outside the bound after fault, record zero. If no qualifying interval before termination, record null (not recovered, right-censored at episode end). Diagnostic only, no added acceptance threshold."},
        "pairing": "Same reset noise/seed, model, PD gains, fault timing and motor; deterministic PPO prediction. Initial qpos hashes must agree. PD is evaluated once and reused for all three pairs.",
        "execution": {"device": "cpu", "torch_threads": 1, "training": False,
                      "rendering_during_evaluation": False, "cuda": False},
        "video_renderer_sha256": digest(Path(r"C:\Users\Zidan\Documents\Codex\work\balance-media\make_media.py")),
        "video": "One silent 30s 1920x1080 25fps video showing PD and all three checkpoints for seed 400. Use recorded 50Hz poses, no replay selection. English serif overlays, no project title. No reselection after results.",
        "failure_policy": "Preserve every result, including failed checkpoints. No tuning, checkpoint change, threshold changes, or video seed reselection."}
    OUT.mkdir(parents=True)
    write(OUT / "protocol.json", protocol)
    sources = ["go2_env.py", "run.py", "improve.py", "extend.py", "sudden.py", "sudden_video.py", "test_sudden.py", "requirements-lock.txt"]
    write(OUT / "design.json", {"protocol_sha256": digest(OUT / "protocol.json"),
                               "source_sha256": {p: digest(ROOT / p) for p in sources},
                               "prior_artifacts_sha256": files})
    print(f"Protocol frozen: seeds {seeds[0]}-{seeds[-1]}, video seed 400, {len(files)} prior artifacts protected", flush=True)


def verify_frozen():
    design = read(OUT / "design.json")
    assert digest(OUT / "protocol.json") == design["protocol_sha256"]
    for path, sha in design["source_sha256"].items():
        assert digest(ROOT / path) == sha, path
    protocol = read(OUT / "protocol.json")
    for checkpoint in protocol["checkpoints"].values():
        assert digest(ROOT / checkpoint["path"]) == checkpoint["sha256"]
    return protocol


def evaluate():
    protocol = verify_frozen()
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                  "torch": torch.__version__, "cuda_build": torch.version.cuda,
                                  "mujoco": mujoco.__version__, "device": "cpu", "training": False})
    video = {}
    for name in protocol["video_controllers"]:
        checkpoint = protocol["checkpoints"].get(name)
        policy = None if checkpoint is None else PPO.load(ROOT / checkpoint["path"], device="cpu")
        if policy is not None:
            assert policy.num_timesteps == 262144
            assert all(torch.isfinite(p).all() for p in policy.policy.parameters())
        env = SuddenBalance(**({} if checkpoint is None else checkpoint["environment"]))
        rows = []
        with (OUT / f"{name}_trace.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["seed", *FIELDS])
            for seed in protocol["evaluation_seeds"]:
                started = time.perf_counter()
                observation, _ = env.reset(seed=seed)
                initial = digest_bytes(env.data.qpos.tobytes())
                poses, tilts = [env.data.qpos.copy()], [env.metrics()["tilt_deg"]]
                for _ in range(env.horizon):
                    action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                    observation, _, terminated, truncated, info = env.step(action)
                    if seed == protocol["video_seed"]:
                        poses.append(env.data.qpos.copy())
                        tilts.append(info["tilt_deg"])
                    if terminated or truncated:
                        break
                assert [r[0] for r in env.samples] == list(range(env.samples[-1][0] + 1))
                writer.writerows((seed, *r) for r in env.samples)
                handle.flush()
                row = {"seed": seed, "initial_state_sha256": initial, **summarize(env.samples),
                       "wall_s": time.perf_counter() - started}
                rows.append(row)
                if seed == protocol["video_seed"]:
                    video[name + "_qpos"] = np.array(poses)
                    video[name + "_tilt"] = np.array(tilts)
                print(f"{name} seed={seed}: post={row['mean_post_tilt_deg']:.3f} peak={row['max_tilt_deg']:.3f} recovery={row['recovery_s']} fallen={row['fallen']}", flush=True)
        env.close()
        write(OUT / f"{name}.json", rows)
    np.savez_compressed(OUT / "video_motion.npz", **video)
    audit()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def audit():
    protocol = verify_frozen()
    rows = {}
    counts = {}
    for name in protocol["video_controllers"]:
        recorded = read(OUT / f"{name}.json")
        assert [r["seed"] for r in recorded] == protocol["evaluation_seeds"]
        grouped = {seed: [] for seed in protocol["evaluation_seeds"]}
        with (OUT / f"{name}_trace.csv").open() as handle:
            for r in csv.DictReader(handle):
                grouped[int(r["seed"])].append([float(r[k]) if k != "fallen" else float(r[k] == "True") for k in FIELDS])
        for row in recorded:
            raw = grouped[row["seed"]]
            assert [int(r[0]) for r in raw] == list(range(len(raw)))
            for r in raw:
                assert r[1] == r[0] * DT
                assert r[5] == (1 if r[0] < ONSET else .1)
                assert r[6] == (1 if r[0] <= ONSET else .1)
            assert summarize(raw) == {k: row[k] for k in summarize(raw)}
        rows[name] = recorded
        counts[name] = sum(map(len, grouped.values()))
    comparisons = {name: criteria(rows["pd"], rows[name]) for name in protocol["checkpoints"]}
    result = {"passed": all(r["passed"] for r in comparisons.values()), "checkpoints": comparisons,
              "raw_sample_counts": counts, "paired_initial_states_verified": True,
              "protocol_sha256": digest(OUT / "protocol.json")}
    path = OUT / "acceptance.json"
    if path.exists():
        assert read(path) == result
    else:
        write(path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate", "audit"])
    {"prepare": prepare, "evaluate": evaluate, "audit": audit}[parser.parse_args().stage]()
