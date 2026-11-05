"""Independent telemetry checks and source-backed summaries for the paired replay study."""
from project_paths import project_file

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import diagnose_actions as d
import review_multimotor as review


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_telemetry(folder, config):
    diagnoses = d.m.s.read(folder / "diagnoses.json")
    episodes, rows = review.verify_dataset(folder, [100, 101, 102, 103, 104])
    data = np.load(folder / "control_telemetry.npz")
    env = d.m.MultiMotorBalance(**config)
    for i, (diag, episode) in enumerate(zip(diagnoses, episodes, strict=True)):
        assert diag["condition"] == episode["condition"] == data["condition"][i]
        assert diag["seed"] == episode["seed"] == data["seed"][i]
        n = int(data["valid_control_steps"][i])
        assert n == diag["control_steps"]
        a = {key: data[key][i, :n] for key in ["start_tick", "actor_raw", "input_action", "applied_action",
             "gate", "target_delta_rad", "joint_target_clipped", "torque_clip_count", "physics_steps", "reward_terms", "reward"]}
        post = a["start_tick"] >= 2500
        assert post.sum() == diag["post_fault_control_steps"]
        np.testing.assert_array_equal(np.clip(a["actor_raw"], -1, 1), a["input_action"])
        np.testing.assert_array_equal(a["input_action"] * a["gate"][:, None], a["applied_action"])
        request = env.home + config["residual_scale"] * a["applied_action"]
        target = np.clip(request, env.model.jnt_range[env.joints, 0], env.model.jnt_range[env.joints, 1])
        np.testing.assert_array_equal(target - env.home, a["target_delta_rad"])
        np.testing.assert_array_equal(request != target, a["joint_target_clipped"])
        assert ((a["torque_clip_count"] >= 0) & (a["torque_clip_count"] <= a["physics_steps"][:, None])).all()
        np.testing.assert_allclose(a["reward_terms"].sum(axis=1), a["reward"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(a["reward_terms"][:, 4], -config["residual_penalty"] * np.sum(a["applied_action"]**2, axis=1), rtol=0, atol=1e-12)
        previous = np.vstack([np.zeros(12), a["applied_action"][:-1]])
        np.testing.assert_allclose(a["reward_terms"][:, 5], -.02 * np.sum((a["applied_action"] - previous)**2, axis=1), rtol=0, atol=1e-12)
        independent = {
            "actor_box_clip_fraction": float(np.mean(np.abs(a["actor_raw"][post]) > 1)),
            "joint_target_clip_fraction": float(np.mean(a["joint_target_clipped"][post])),
            "torque_clip_fraction": float(a["torque_clip_count"][post].sum() / (12 * a["physics_steps"][post].sum())),
            "gate_off_fraction": float(np.mean(a["gate"][post] == 0)),
            "target_delta_rms_rad": float(np.sqrt(np.mean(a["target_delta_rad"][post]**2))),
            "full_return": float(a["reward"].sum()), "post_fault_return": float(a["reward"][post].sum())}
        for key, value in independent.items():
            np.testing.assert_allclose(value, diag[key], rtol=0, atol=1e-12)
        for index, key in enumerate(d.REWARD_FIELDS):
            np.testing.assert_allclose(a["reward_terms"][post, index].mean(), diag["mean_reward_terms_after_fault"][key], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a["target_delta_rad"][post].mean(axis=0), diag["target_delta_mean_rad_by_joint"])
        np.testing.assert_array_equal(a["torque_clip_count"][post].sum(axis=0) / a["physics_steps"][post].sum(), diag["torque_clip_fraction_by_joint"])
        assert diag["max_torque_reconstruction_error"] <= 1e-10 and diag["max_reward_reconstruction_error"] <= 1e-10
    env.close()
    data.close()
    return episodes, diagnoses, rows


def main():
    p = d.verify()
    assert "Ran 36 tests" in (d.OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    items = p["checkpoints"]
    joints = d.m.s.read(d.OUT / "schema.json")["joint_order"]
    episodes_by_label, groups, episode_rows, joint_rows = {}, [], [], []
    total_rows, replay_errors = 0, []
    for label, item in items.items():
        folder = d.OUT / label
        episodes, diagnostics, count = verify_telemetry(folder, p["environment"])
        episodes_by_label[label] = episodes
        total_rows += count
        proof = d.m.s.read(folder / "replay_verification.json")
        assert proof["matched_to_previous_data"] and len(proof["episodes"]) == 25
        replay_errors.extend(row["max_metric_error"] for row in proof["episodes"])
        for episode, diag in zip(episodes, diagnostics, strict=True):
            episode_rows.append({"checkpoint": label, "budget": item["budget"], "training_seed": item["training_seed"],
                "condition": diag["condition"], "evaluation_seed": diag["seed"],
                "mean_post_tilt_deg": episode["mean_post_tilt_deg"], "peak_tilt_deg": diag["peak_tilt_deg"],
                "peak_time_s": diag["peak_time_s"], "late_mean_tilt_deg": diag["late_mean_tilt_deg"],
                "correction_rms_rad": diag["target_delta_rms_rad"], "actor_clip_fraction": diag["actor_box_clip_fraction"],
                "target_clip_fraction": diag["joint_target_clip_fraction"], "torque_clip_fraction": diag["torque_clip_fraction"],
                "gate_off_fraction": diag["gate_off_fraction"], "full_return": diag["full_return"],
                **{f"reward_{key}": value for key, value in diag["mean_reward_terms_after_fault"].items()}})
            for j, name in enumerate(joints):
                joint_rows.append({"checkpoint": label, "condition": diag["condition"], "evaluation_seed": diag["seed"],
                    "joint": name, "mean_target_delta_rad": diag["target_delta_mean_rad_by_joint"][j],
                    "rms_target_delta_rad": diag["target_delta_rms_rad_by_joint"][j],
                    "actor_clip_fraction": diag["actor_clip_fraction_by_joint"][j],
                    "torque_clip_fraction": diag["torque_clip_fraction_by_joint"][j],
                    "mean_abs_delivered_torque_nm": diag["delivered_abs_torque_by_joint"][j]})
        for condition in d.m.CONDITIONS:
            selected = [row for row in episode_rows if row["checkpoint"] == label and row["condition"] == condition]
            ds = [row for row in diagnostics if row["condition"] == condition]
            groups.append({"checkpoint": label, "budget": item["budget"], "training_seed": item["training_seed"],
                "condition": condition, **{key: float(np.mean([row[key] for row in selected])) for key in selected[0]
                  if key not in ["checkpoint", "budget", "training_seed", "condition", "evaluation_seed"]},
                "mean_target_delta_rad_by_joint": np.mean([row["target_delta_mean_rad_by_joint"] for row in ds], axis=0).tolist()})
    write_csv(d.OUT / "episode_summary.csv", episode_rows)
    write_csv(d.OUT / "joint_summary.csv", joint_rows)
    d.m.s.write(d.OUT / "diagnostic_summary.json", {"groups": groups, "episodes": len(episode_rows),
        "matched_physics_rows": total_rows, "max_replay_metric_error": max(replay_errors),
        "telemetry_aggregates_independently_verified": True})
    baseline = episodes_by_label["pd"]
    comparisons = {label: d.m.evidence(baseline, episodes, p["validation_seeds"])
                   for label, episodes in episodes_by_label.items() if label != "pd"}
    d.m.s.write(d.OUT / "comparisons.json", comparisons)
    heat_rows = [g for g in groups if g["budget"] == 1048576 and g["condition"] != "healthy"]
    heat = np.array([g["mean_target_delta_rad_by_joint"] for g in heat_rows])
    font = Path(r"C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "font.size": 11})
    fig, (ax, costs_ax) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.5, 1]})
    limit = max(.01, float(np.abs(heat).max()))
    im = ax.imshow(heat, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    names = [f"Model {g['training_seed'] + 1} / {g['condition'].replace('_calf', '')}" for g in heat_rows]
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(12), [name.replace('_', '\n') for name in joints], fontsize=9)
    ax.set_title("Mean joint-target corrections after the fault")
    fig.colorbar(im, ax=ax, fraction=.045, pad=.02, label="Joint-target change (rad)")
    start = np.zeros(len(heat_rows))
    for term, color in [("tilt", "#355c7d"), ("height", "#859f75"), ("xy", "#bf9f68"), ("action", "#aa6a54"), ("smoothness", "#9b85a7")]:
        values = np.array([-g[f"reward_{term}"] for g in heat_rows])
        costs_ax.barh(range(len(heat_rows)), values, left=start, label=term, color=color)
        start += values
    costs_ax.set_yticks(range(len(names)), names)
    costs_ax.invert_yaxis()
    costs_ax.set_xlabel("Mean penalty per control step")
    costs_ax.set_title("Reward costs at the maximum training budget")
    costs_ax.legend(fontsize=9)
    fig.suptitle("Which corrections and reward costs accompany the remaining tilt?", fontsize=20)
    fig.text(.5, .02, "Five matched validation starts per row. Signed target changes do not establish beneficial direction. Associations are not isolated causes.", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .05, 1, .94])
    fig.savefig(d.OUT / "diagnosis.png", dpi=150)
    plt.close(fig)
    for name, sha in d.m.s.read(d.OUT / "design.json")["protected_artifacts_sha256"].items():
        assert d.m.s.digest(project_file(d.m.s.ROOT, name)) == sha, name
    print("Independent diagnostic verification complete:", len(episode_rows), "episodes;", total_rows, "matching physics rows", flush=True)


if __name__ == "__main__":
    main()
