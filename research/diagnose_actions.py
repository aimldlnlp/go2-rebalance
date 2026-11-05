"""Read-only replay diagnostics for action clipping, motor effort and reward costs."""
from project_paths import project_file

import argparse
import csv
from datetime import datetime, timezone
import json
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

import multimotor as m
import multimotor_extend as extension

OUT = m.s.ROOT / "results/action_diagnosis"
REWARD_FIELDS = ["alive", "tilt", "height", "xy", "action", "smoothness", "fall"]


def checkpoints():
    items = {"pd": {"path": None, "reference": str(m.OUT / "validation/pd"), "budget": 0, "training_seed": -1}}
    for budget in [262144, 524288, 1048576]:
        parents = extension.initial_parents() if budget == 262144 else {
            seed: extension.OUT / f"round_{budget}/seed_{seed}" for seed in range(3)}
        for seed, parent in parents.items():
            items[f"b{budget}_s{seed}"] = {"path": str(parent / "policy.zip"),
                "reference": str(parent / "validation"), "budget": budget, "training_seed": seed}
    return items


class Recorder:
    def __init__(self, env):
        self.env = env
        self.original_step = mujoco.mj_step
        self.rows = []
        self.max_torque_check_error = 0.0
        self.max_reward_check_error = 0.0

    def physics_step(self, model, data):
        assert model is self.env.model and data is self.env.data
        raw = 50 * (self.target - data.qpos[self.env.qadr]) - 3 * data.qvel[self.env.vadr]
        expected = np.clip(raw, self.env.limits[:, 0], self.env.limits[:, 1])
        expected[self.env.weak_motor] *= self.env.motor_strength()
        error = float(np.max(np.abs(expected - data.ctrl)))
        assert error <= 1e-10
        self.max_torque_check_error = max(self.max_torque_check_error, error)
        self.count += 1
        self.clips += (raw < self.env.limits[:, 0]) | (raw > self.env.limits[:, 1])
        self.requested += np.abs(raw)
        self.delivered += np.abs(data.ctrl)
        self.torque_max = np.maximum(self.torque_max, np.abs(data.ctrl))
        self.original_step(model, data)

    def step(self, action, actor_raw):
        env = self.env
        action, actor_raw = np.asarray(action, dtype=float), np.asarray(actor_raw, dtype=float)
        if action.shape != (12,) or actor_raw.shape != (12,) or not np.isfinite(action).all() or not np.isfinite(actor_raw).all():
            raise ValueError("Expected two finite 12-value action vectors")
        clipped = np.clip(action, -1, 1)
        gate = float(np.clip(env.metrics()["tilt_deg"] - env.tilt_deadband_deg, 0, 1)) if env.tilt_deadband_deg is not None else 1.0
        applied = clipped * gate
        previous = env.previous.copy()
        requested_target = env.home + env.residual_scale * applied
        self.target = np.clip(requested_target, env.model.jnt_range[env.joints, 0], env.model.jnt_range[env.joints, 1])
        start_tick = round(env.data.time / m.s.DT)
        joint_position, joint_velocity = env.data.qpos[env.qadr].copy(), env.data.qvel[env.vadr].copy()
        self.count = 0
        self.clips, self.requested, self.delivered, self.torque_max = [np.zeros(12) for _ in range(4)]
        with patch.object(mujoco, "mj_step", new=self.physics_step):
            observation, reward, terminated, truncated, info = env.step(action)
        assert self.count > 0
        terms = np.array([1.0, -env.tilt_weight * np.deg2rad(info["tilt_deg"])**2,
            -10 * (info["height_m"] - .27)**2, -2 * float(env.data.qpos[:2] @ env.data.qpos[:2]),
            -env.residual_penalty * float(applied @ applied), -.02 * float((applied - previous) @ (applied - previous)),
            -10.0 if info["fallen"] else 0.0])
        error = abs(float(terms.sum()) - reward)
        assert error <= 1e-10
        self.max_reward_check_error = max(self.max_reward_check_error, error)
        rotation = env.data.xmat[env.base].reshape(3, 3)
        angles = np.rad2deg([np.arctan2(rotation[2, 1], rotation[2, 2]),
                            np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))])
        self.rows.append({"start_tick": start_tick, "time_s": info["time_s"], "actor_raw": actor_raw.copy(),
            "input_action": clipped, "applied_action": applied, "gate": gate,
            "target_delta_rad": self.target - env.home,
            "joint_target_clipped": requested_target != self.target,
            "joint_position": joint_position, "joint_velocity": joint_velocity,
            "torque_clip_count": self.clips.copy(), "physics_steps": self.count,
            "requested_abs_torque": self.requested / self.count,
            "delivered_abs_torque": self.delivered / self.count, "delivered_max_abs_torque": self.torque_max.copy(),
            "reward_terms": terms, "reward": reward, "tilt_deg": info["tilt_deg"],
            "roll_pitch_deg": angles, "angular_velocity": env.data.qvel[3:6].copy()})
        return observation, reward, terminated, truncated, info

    def arrays(self):
        return {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}

    def summary(self):
        a = self.arrays()
        post = a["start_tick"] >= 2500
        assert post.any()
        nphysics = float(a["physics_steps"][post].sum())
        peak = max(self.env.samples, key=lambda row: row[2])
        return {"control_steps": len(a["time_s"]), "post_fault_control_steps": int(post.sum()),
            "gate_off_fraction": float(np.mean(a["gate"][post] == 0)),
            "actor_box_clip_fraction": float(np.mean(np.abs(a["actor_raw"][post]) > 1)),
            "joint_target_clip_fraction": float(np.mean(a["joint_target_clipped"][post])),
            "target_delta_rms_rad": float(np.sqrt(np.mean(a["target_delta_rad"][post]**2))),
            "target_delta_mean_rad_by_joint": a["target_delta_rad"][post].mean(axis=0).tolist(),
            "target_delta_rms_rad_by_joint": np.sqrt(np.mean(a["target_delta_rad"][post]**2, axis=0)).tolist(),
            "actor_clip_fraction_by_joint": np.mean(np.abs(a["actor_raw"][post]) > 1, axis=0).tolist(),
            "torque_clip_fraction_by_joint": (a["torque_clip_count"][post].sum(axis=0) / nphysics).tolist(),
            "torque_clip_fraction": float(a["torque_clip_count"][post].sum() / (12 * nphysics)),
            "delivered_abs_torque_by_joint": a["delivered_abs_torque"][post].mean(axis=0).tolist(),
            "mean_reward_terms_after_fault": dict(zip(REWARD_FIELDS, a["reward_terms"][post].mean(axis=0).tolist(), strict=True)),
            "post_fault_return": float(a["reward"][post].sum()), "full_return": float(a["reward"].sum()),
            "peak_time_s": float(peak[1]), "peak_tilt_deg": float(peak[2]),
            "late_mean_tilt_deg": float(np.mean([row[2] for row in self.env.samples if row[0] > 12500])),
            "max_torque_reconstruction_error": self.max_torque_check_error,
            "max_reward_reconstruction_error": self.max_reward_check_error}


def actor_action(policy, observation):
    if policy is None:
        return np.zeros(12), np.zeros(12)
    action = policy.predict(observation, deterministic=True)[0]
    assert not policy.policy.squash_output
    tensor, _ = policy.policy.obs_to_tensor(observation)
    with torch.no_grad():
        raw = policy.policy._predict(tensor, deterministic=True).cpu().numpy()[0]
    np.testing.assert_array_equal(np.clip(raw, -1, 1), action)
    return action, raw


def source_traces(folder):
    rows = {}
    with (folder / "trace.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = row["condition"], int(row["seed"])
            rows.setdefault(key, []).append([float(row[k]) if k != "fallen" else float(row[k] == "True") for k in m.s.FIELDS])
    return rows


def replay(label, policy, folder, config, seeds, reference=None):
    folder.mkdir(parents=True)
    expected = source_traces(reference) if reference is not None else None
    expected_rows = {(r["condition"], r["seed"]): r for r in m.s.read(reference / "episodes.json")} if reference is not None else None
    episodes, diagnoses, tensors, proofs = [], [], [], []
    with (folder / "trace.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", *m.s.FIELDS])
        for condition in m.CONDITIONS:
            env = m.MultiMotorBalance(motor=condition, record=True, **config)
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                initial = m.s.digest_bytes(env.data.qpos.tobytes())
                recorder = Recorder(env)
                for _ in range(env.horizon):
                    action, raw = actor_action(policy, observation)
                    observation, _, terminated, truncated, _ = recorder.step(action, raw)
                    if terminated or truncated:
                        break
                assert env.samples[-1][0] > 2500
                metrics = {"condition": condition, "seed": seed, "initial_state_sha256": initial,
                    **m.s.summarize(env.samples), "mean_tilt_deg": float(np.mean([row[2] for row in env.samples[1:]]))}
                if expected is not None:
                    original = np.asarray(expected[condition, seed])
                    actual = np.asarray(env.samples, dtype=float)
                    np.testing.assert_allclose(actual, original, atol=1e-10, rtol=0)
                    assert initial == expected_rows[condition, seed]["initial_state_sha256"]
                    proofs.append({"condition": condition, "seed": seed, "physics_samples": len(actual),
                                   "max_metric_error": float(np.max(np.abs(actual - original)))})
                writer.writerows((condition, seed, *row) for row in env.samples)
                handle.flush()
                episodes.append(metrics)
                diagnoses.append({"condition": condition, "seed": seed, **recorder.summary()})
                tensors.append(recorder.arrays())
            env.close()
            group = diagnoses[-len(seeds):]
            print(f"{label}/{condition}: actor_clip={np.mean([r['actor_box_clip_fraction'] for r in group]):.3f}, torque_clip={np.mean([r['torque_clip_fraction'] for r in group]):.3f}", flush=True)
    packed = {"valid_control_steps": np.array([len(item["time_s"]) for item in tensors]),
              "condition": np.array([row["condition"] for row in episodes]), "seed": np.array([row["seed"] for row in episodes])}
    for key in tensors[0]:
        array = np.zeros((len(tensors), 1500, *tensors[0][key].shape[1:]), dtype=tensors[0][key].dtype)
        for i, item in enumerate(tensors):
            array[i, :len(item[key])] = item[key]
        packed[key] = array
    np.savez_compressed(folder / "control_telemetry.npz", **packed)
    m.s.write(folder / "episodes.json", episodes)
    m.s.write(folder / "diagnoses.json", diagnoses)
    m.s.write(folder / "replay_verification.json", {"matched_to_previous_data": reference is not None, "episodes": proofs})
    return episodes, diagnoses


def prepare():
    assert not OUT.exists()
    extension.verify()
    protected = {}
    for base in [m.OUT, extension.OUT]:
        for name, sha in m.s.read(base / "evidence_manifest.json").items():
            assert m.s.digest(base / name) == sha, name
            protected[str((base / name).relative_to(m.s.ROOT))] = sha
    items = checkpoints()
    for item in items.values():
        if item["path"] is not None:
            item["sha256"] = m.s.digest(item["path"])
    p = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": items,
        "validation_seeds": [100, 101, 102, 103, 104], "reserved_seeds_unused": list(range(700, 720)),
        "conditions": m.CONDITIONS, "environment": m.verify()["configurations"]["B"],
        "episodes": 250, "training": False, "cuda": False, "rendering": False,
        "method": "Replay PD plus all nine B checkpoints on the same five old validation seeds and all five conditions. Hook immediately before each native physics step without changing the environment implementation. Reconstruct and assert actual clipped/strength-scaled torques, log per-joint saturation, and verify exact reward decomposition. Match every physics sample against stored traces. Read the raw Gaussian actor mean separately and assert its clipped value equals the public deterministic action.",
        "definitions": {"actor_clip": "Fraction of 12 raw actor outputs strictly outside [-1,1] before the environment gate, after fault time.",
            "joint_target_clip": "Fraction of target values changed by joint-range clipping.",
            "torque_clip": "Fraction of joint/physics-step commands outside original actuator limits before the selected motor's strength multiplier.",
            "reward": "Exact signed components: alive, tilt, height, xy, applied-action regularization, action-change regularization, fall. Post-fault comparison uses control intervals starting at tick >=2500.",
            "telemetry": "Per-controller NPZ arrays are padded after termination; valid_control_steps is mandatory. Initial state and full 500 Hz traces remain in trace.csv and episodes.json. Joint names are saved in schema.json."},
        "interpretation": "Clipping and reward contributions are observational evidence, not isolated causal mechanisms. After inspecting the diagnostics, freeze exactly one small controlled intervention before testing it on old validation seeds. Do not train or test holdout seeds during diagnosis, select a best-looking example, loosen acceptance criteria, or alter earlier artifacts."}
    OUT.mkdir()
    m.s.write(OUT / "protocol.json", p)
    m.s.write(OUT / "design.json", {"protocol_sha256": m.s.digest(OUT / "protocol.json"),
        "source_sha256": {name: m.s.digest(project_file(m.s.ROOT, name)) for name in ["diagnose_actions.py", "test_diagnose_actions.py", "go2_env.py", "multimotor.py"]},
        "protected_artifacts_sha256": protected})
    env = m.MultiMotorBalance()
    m.s.write(OUT / "schema.json", {"joint_order": [env.model.actuator(i).name for i in range(12)], "reward_term_order": REWARD_FIELDS})
    env.close()
    print("Diagnosis protocol frozen: 250 matched replay episodes, no training or holdout use", flush=True)


def verify():
    p, design = m.s.read(OUT / "protocol.json"), m.s.read(OUT / "design.json")
    assert m.s.digest(OUT / "protocol.json") == design["protocol_sha256"]
    for name, sha in design["source_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    return p


def run():
    p = verify()
    m.s.write(OUT / "execution.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "training": False,
                                      "device": "cpu", "cuda_build": torch.version.cuda})
    for label, item in p["checkpoints"].items():
        policy = None if item["path"] is None else PPO.load(item["path"], device="cpu")
        replay(label, policy, OUT / label, p["environment"], p["validation_seeds"], m.s.ROOT / item["reference"])
        if policy is not None:
            assert m.s.digest(item["path"]) == item["sha256"]
    for name, sha in m.s.read(OUT / "design.json")["protected_artifacts_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    print("All diagnostic replays complete and previous artifacts unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    {"prepare": prepare, "run": run}[parser.parse_args().stage]()
