"""Finalize the observed diagnosis and the one controlled, non-training intervention."""
from project_paths import project_file

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

import diagnose_actions as d
import review_action_diagnosis as audit
import review_multimotor as metrics


def main():
    p = d.verify()
    study = d.m.s.read(d.OUT / "diagnostic_summary.json")
    comparison = d.m.s.read(d.OUT / "gain_ablation/comparison.json")
    plan = d.m.s.read(d.OUT / "gain_ablation/protocol.json")
    assert study["episodes"] == 250 and study["matched_physics_rows"] == 3750250
    assert study["max_replay_metric_error"] == 0
    assert plan["changed_parameter"] == "residual_scale" and plan["original_value"] == .35 and plan["intervention_value"] == .25
    assert {key for key in plan["environment"] if plan["environment"][key] != p["environment"][key]} == {"residual_scale"}
    assert not plan["training"] and not comparison["final_seeds_used"]
    baseline = d.m.s.read(d.OUT / "pd/episodes.json")
    new_rows = 0
    for seed in range(3):
        key = str(seed)
        path = d.OUT / f"gain_ablation/seed_{seed}"
        episodes, _, count = audit.verify_telemetry(path, plan["environment"])
        new_rows += count
        independent = metrics.independent_comparison(baseline, episodes, p["validation_seeds"])
        metrics.assert_comparison(independent, comparison["training_seeds"][key]["intervention"])
        assert d.m.s.digest(plan["policy_checkpoints"][key]) == plan["checkpoint_sha256"][key]
    original_means = new_means = original_peaks = new_peaks = improved = regressed = 0
    table = []
    for seed, result in comparison["training_seeds"].items():
        for row in result["differences"]:
            if row["condition"] == "healthy":
                assert row["mean_delta_deg"] == 0
                continue
            improved += row["mean_delta_deg"] < 0
            regressed += row["mean_delta_deg"] > 0
            original_means += row["original_reduction_pct"] >= 40
            new_means += row["intervention_reduction_pct"] >= 40
            original_peaks += row["original_peak_passes"]
            new_peaks += row["intervention_peak_passes"]
            table.append(f"| {seed} | {row['condition']} | {row['original_mean_deg']:.4f} | {row['intervention_mean_deg']:.4f} | {row['mean_delta_deg']:+.4f} | {row['intervention_reduction_pct']:.2f}% | {row['original_peak_passes']}/5 → {row['intervention_peak_passes']}/5 |")
    assert (improved, regressed, original_means, new_means, original_peaks, new_peaks) == (2, 10, 5, 4, 55, 54)
    assert not comparison["passed"] and new_rows == 1125075
    assert all(r["original"]["checks"]["survival"] and r["intervention"]["checks"]["survival"] and
               r["intervention"]["checks"]["healthy"] for r in comparison["training_seeds"].values())
    groups = study["groups"]
    fault = [g for g in groups if g["budget"] > 0 and g["condition"] != "healthy"]
    total_penalty = sum(sum(-g["reward_" + key] for key in ["tilt", "height", "xy", "action", "smoothness"]) for g in fault)
    shares = {key: sum(-g["reward_" + key] for g in fault) / total_penalty for key in ["tilt", "height", "xy", "action", "smoothness"]}
    assert max(g["target_clip_fraction"] for g in fault) == 0
    early = d.m.s.read(d.OUT / "b524288_s0/diagnoses.json")
    early = [r for r in early if r["condition"] == "FL_calf"]
    hip_clip = float(np.mean([r["actor_clip_fraction_by_joint"][0] for r in early]))
    knee_clip = float(np.mean([r["torque_clip_fraction_by_joint"][2] for r in early]))
    latest = d.m.s.read(d.OUT / "b1048576_s2/diagnoses.json")
    candidate = d.m.s.read(d.OUT / "gain_ablation/seed_2/diagnoses.json")
    saturation_changes = {}
    for condition, index in [("FR_calf", 5), ("RR_calf", 11)]:
        before = [r for r in latest if r["condition"] == condition]
        after = [r for r in candidate if r["condition"] == condition]
        saturation_changes[condition] = {"old_weak_knee_torque_clip_fraction": float(np.mean([r["torque_clip_fraction_by_joint"][index] for r in before])),
            "new_weak_knee_torque_clip_fraction": float(np.mean([r["torque_clip_fraction_by_joint"][index] for r in after]))}
    font = Path(r"C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "font.size": 12})
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
    titles = ["Front left", "Front right", "Rear left", "Rear right"]
    for i, (condition, ax) in enumerate(zip(d.m.MOTORS, axes, strict=True)):
        for seed in range(3):
            row = next(r for r in comparison["training_seeds"][str(seed)]["differences"] if r["condition"] == condition)
            ax.plot([row["original_mean_deg"], row["intervention_mean_deg"]], [seed, seed], color="#b9bbb7", linewidth=2)
            ax.scatter(row["original_mean_deg"], seed, color="#355c7d", marker="o", s=60, zorder=3)
            ax.scatter(row["intervention_mean_deg"], seed, color="#9a6544", marker="s", s=50, zorder=3)
        reference = comparison["training_seeds"]["0"]["original"]["groups"][i + 1]["pd_mean_deg"]
        ax.axvline(reference, color="#777777", linestyle=":", linewidth=1)
        ax.axvline(.6 * reference, color="black", linestyle="--", linewidth=1)
        ax.set_title(titles[i])
        ax.set_yticks(range(3), ["Model 1", "Model 2", "Model 3"])
        ax.set_xlabel("Mean body tilt after fault (°)")
        ax.set_xlim(0, 18)
        ax.set_ylim(2.5, -.5)
    legend = [Line2D([0], [0], marker="o", color="#355c7d", linestyle="None", label="Original coefficient: 0.35"),
              Line2D([0], [0], marker="s", color="#9a6544", linestyle="None", label="Tested coefficient: 0.25"),
              Line2D([0], [0], color="black", linestyle="--", label="Required mean: 60% of reference"),
              Line2D([0], [0], color="#777777", linestyle=":", label="Fixed-target reference")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .88), ncol=2, fontsize=10)
    fig.suptitle("Reducing correction amplitude helps two cases and worsens ten", fontsize=20)
    fig.text(.5, .02, "Same frozen policies and five matched validation starts per case. Lower tilt is better. Peaks and survival are evaluated separately.", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .06, 1, .7])
    fig.savefig(d.OUT / "gain_ablation.png", dpi=150)
    plt.close(fig)
    text = f"""# Joint-action diagnosis and one controlled intervention

**Diagnosis complete; the tested change is rejected as a universal fix.** PD and all nine B checkpoints were replayed on identical old validation conditions. Every one of {study['matched_physics_rows']:,} physics samples matched the original traces with zero numerical error. The subsequent single-parameter test kept all three latest policy checkpoints fixed and reduced their target-correction coefficient from 0.35 to 0.25. It improved two motor/model means and worsened ten; none of the three modified controllers passed all criteria.

![Joint corrections and reward costs](diagnosis.png)

## What the measurements show

No joint-range target clipping occurred in the 180 diagnosed learned-policy fault episodes. Action clipping and torque clipping appeared in specific combinations rather than all failures. At 524,288 steps, training seed 0 under front-left weakness clipped the raw FL hip action in {100 * hip_clip:.2f}% of post-fault control steps, while the weak FL knee reached the nominal torque bound in {100 * knee_clip:.2f}% of post-fault physics steps. These are different joints and different clipping stages. A global saturation average hides that distinction.

At the maximum budget, seed 2's right-side failures had higher correction RMS (about 0.177 and 0.183 rad) than its left-side cases (about 0.117 and 0.118 rad). Mean tilt in the final five seconds remained about 16.17° for the front-right fault and 15.93° for the rear-right fault. The rear-right maximum occurred at 30 seconds on average. These measurements describe persistent stance error as well as early transients; the evidence does not support treating every failure as a short overshoot.

Across all nine checkpoints and four fault types, tilt contributed {100 * shares['tilt']:.2f}% of the mean non-fall penalties, height {100 * shares['height']:.2f}%, horizontal position {100 * shares['xy']:.2f}%, applied action {100 * shares['action']:.2f}%, and action changes {100 * shares['smoothness']:.4f}%. This weights equally the 180 full fault episodes and their 1,250 post-fault control intervals. The constant alive reward is excluded from the penalty denominator; fall penalties are zero. The reward already penalizes tilt strongly in these runs. There is no explicit per-episode PD-relative peak constraint or 40% per-motor guarantee in the scalar reward. That structural difference is not proof that changing a penalty alone will solve the learning problem.

The signed correction map shows which joint targets move together in each case. A signed angle does not establish that a joint is pushing the wrong way. Joint coupling, closed-loop feedback and geometry prevent assigning an individual joint as the cause from this observational comparison alone.

## Single frozen test

The intervention was selected after diagnosis and frozen before its first rollout in `gain_ablation/protocol.json`. It changes only `residual_scale`, from 0.35 to 0.25 rad per unit applied action, on the three existing 1,048,576-step policies. The reduced coefficient is 28.57% lower; actual closed-loop corrections need not decrease by that fraction because the observations and actions change with the resulting motion. Reward weights, PD gains, gates, joint ranges, torque limits, model, noise, fault timing, motor strength, weights and evaluation thresholds remain unchanged. The policies were trained with 0.35; this is an inference-time calibration experiment, not a newly trained policy.

![Controlled gain comparison](gain_ablation.png)

| Training seed | Fault | Original mean (°) | Tested mean (°) | Change (°) | Tested reduction vs PD | Paired peak passes |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

All 75 intervention episodes completed 30 seconds without a fall. Healthy tilt stayed exactly equal to PD. The number of motor/model combinations meeting the unchanged 40% mean-reduction target fell from 5/12 to 4/12. Paired peak passes fell from 55/60 to 54/60. No checkpoint passed the full set. The original 0.35 coefficient remains the project's unchanged default; the unsuccessful candidate was not adopted and no second value was searched.

For maximum-budget seed 2, weaker correction amplitude improved the two poor right-side means from 16.1790° to 13.9732° and from 15.2739° to 12.8785°. Both still missed the 40% target. The weak-knee torque clipping rates changed from {100 * saturation_changes['FR_calf']['old_weak_knee_torque_clip_fraction']:.2f}% to {100 * saturation_changes['FR_calf']['new_weak_knee_torque_clip_fraction']:.2f}% on the front-right fault and from {100 * saturation_changes['RR_calf']['old_weak_knee_torque_clip_fraction']:.2f}% to {100 * saturation_changes['RR_calf']['new_weak_knee_torque_clip_fraction']:.2f}% on the rear-right fault. Reducing clipping therefore did not deliver the required overall performance in this tested controller. The gain change affects the entire closed loop, so this is not an isolated intervention on clipping itself.

## Scope and verification

Diagnosis covers healthy operation and four separate 10%-strength knee-motor faults at 5 seconds, with evaluation seeds 100–104, at budgets 262,144, 524,288 and 1,048,576 for training seeds 0–2, plus PD. The one intervention uses only the latest three models on those same five starts. All 325 replay/intervention episodes ran locally without new policy training, CUDA or simulation rendering. The reserved final seeds 700–719 remain unused. These repeated old validation starts are suitable for diagnosis, not evidence of held-out acceptance.

Thirty-six functional/regression tests passed before replay. Instrumentation wraps only the native physics-step entry, reads pre-step commands and states, asserts reconstructed clipped/strength-scaled commands equal actual controls, then calls the original step. It does not replace controller or environment dynamics. Tests compare exact positions, velocities, physics metrics and scalar rewards with an uninstrumented environment. Raw actor means are read separately and their clipped values must equal the public deterministic action on every control step. Motor gears and gains are 1 and bias terms are zero in this model, so the recorded motor-control commands have the same numerical scale as joint torque.

Reward components are evaluated at 50 Hz, exactly as the environment returns reward; peak tilt and clipping are checked at 500 Hz. Independent checks verify raw measurement means/peaks/recovery, action processing, target bounds, clipping denominators, reward sums and regularization terms. Every diagnostic physics trajectory matches its earlier source trace. The 75 intervention episodes deliberately differ from those traces and were independently verified as new measurements.

`episode_summary.csv` gives episode-level costs, correction sizes, clipping fractions and peak times. `joint_summary.csv` gives all 3,000 episode/joint rows. Each checkpoint folder retains `control_telemetry.npz`, full `trace.csv`, episode outcomes, diagnostics and replay proof. NPZ rows beyond `valid_control_steps` are padding and must be excluded. The gain-test folders retain equivalent raw telemetry and all failures. Protocols, scripts, tests, checkpoint hashes and artifact hashes remain inspectable.

## Interpretation and limits

The evidence establishes uneven closed-loop compensation across motors and training seeds, selective saturation, persistent errors in some regressions, and a failed global gain reduction. It does not isolate one incorrect joint command or prove that reward weights, observations, model capacity, optimizer noise or controller structure are the sole cause. The existing quadratic tilt cost dominates average penalties here; replacing it with a larger number is not justified as an established remedy. A future training intervention would need its own frozen, single-change comparison and all-motor evaluation.

No fresh-seed generalization, walking, simultaneous failures, other joint types, physical hardware, or video was tested or produced in this diagnostic task. Original checkpoints and prior experiment records are unchanged.
"""
    (d.OUT / "REPORT.md").write_text(text, encoding="utf-8")
    protected = dict(d.m.s.read(d.extension.OUT / "design.json")["prior_artifacts_sha256"])
    protected.update(d.m.s.read(d.OUT / "design.json")["protected_artifacts_sha256"])
    for name, sha in protected.items():
        assert d.m.s.digest(project_file(d.m.s.ROOT, name)) == sha, name
    assert not (d.extension.OUT / "final").exists()
    d.m.s.write(d.OUT / "completion_audit.json", {"completed": True, "diagnosis_episodes": 250,
        "original_physics_samples_matched": 3750250, "replay_max_error": 0.0,
        "intervention_episodes": 75, "intervention_physics_samples": new_rows,
        "mean_cases_improved": improved, "mean_cases_regressed": regressed,
        "original_mean_target_passes": original_means, "intervention_mean_target_passes": new_means,
        "original_peak_pair_passes": original_peaks, "intervention_peak_pair_passes": new_peaks,
        "intervention_accepted": False, "default_controller_changed": False, "new_policy_training": False,
        "final_seeds_used": False, "previous_artifacts_unchanged": len(protected), "tests_passed": 36,
        "reward_cost_shares": shares, "weak_knee_saturation_changes": saturation_changes,
        "review_source_sha256": d.m.s.digest(__file__), "cuda": False, "rendering": False})
    print("Diagnosis and the single controlled intervention are complete. Intervention rejected.", flush=True)


if __name__ == "__main__":
    main()
