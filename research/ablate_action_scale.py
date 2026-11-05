"""One frozen controller-gain intervention on the three latest policies; no retraining."""
from project_paths import project_file

from datetime import datetime, timezone

from stable_baselines3 import PPO

import diagnose_actions as d
import review_action_diagnosis as audit
import review_multimotor as metrics


def main():
    p = d.verify()
    summary = d.m.s.read(d.OUT / "diagnostic_summary.json")
    assert summary["episodes"] == 250 and summary["max_replay_metric_error"] <= 1e-10
    folder = d.OUT / "gain_ablation"
    assert not folder.exists(), "Refusing to overwrite the single intervention"
    config = dict(p["environment"], residual_scale=.25)
    assert {key for key in config if config[key] != p["environment"][key]} == {"residual_scale"}
    parents = {str(seed): d.extension.OUT / f"round_1048576/seed_{seed}/policy.zip" for seed in range(3)}
    hashes = {seed: d.m.s.digest(path) for seed, path in parents.items()}
    folder.mkdir()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "changed_parameter": "residual_scale",
        "original_value": .35, "intervention_value": .25, "environment": config,
        "validation_seeds": p["validation_seeds"], "conditions": p["conditions"],
        "policy_checkpoints": {seed: str(path) for seed, path in parents.items()}, "checkpoint_sha256": hashes,
        "criteria": d.m.verify()["criteria"], "training": False, "reserved_final_seeds_used": False,
        "selection_basis": "Latest seed-2 policies have persistently high tilt in right-side faults, larger joint-target RMS changes, and frequent weak-knee torque clipping. Other failures do not clip. Lowering the correction coefficient tests the effect of correction amplitude; it does not assume clipping is the sole cause or guarantee smaller closed-loop corrections.",
        "method": "One intervention only, chosen after diagnosis and frozen before replay. Compare 0.35 versus 0.25 using identical fixed weights, original PD, all five conditions, all three maximum-budget models and seeds 100-104. No new training, parameter search, best-case selection, criteria changes, or final-seed evaluation. This is an inference-time calibration ablation of policies trained with 0.35, not a newly trained policy.",
        "source_sha256": d.m.s.digest(__file__), "diagnostic_source_sha256": d.m.s.digest(d.m.s.ROOT / "research/diagnose_actions.py")}
    d.m.s.write(folder / "protocol.json", plan)
    baseline = d.m.s.read(d.OUT / "pd/episodes.json")
    results = {}
    for seed, path in parents.items():
        policy = PPO.load(path, device="cpu")
        target = folder / f"seed_{seed}"
        episodes, _ = d.replay(f"gain025_s{seed}", policy, target, config, p["validation_seeds"])
        verified, _, _ = audit.verify_telemetry(target, config)
        candidate = d.m.evidence(baseline, episodes, p["validation_seeds"])
        metrics.assert_comparison(metrics.independent_comparison(baseline, verified, p["validation_seeds"]), candidate)
        original = d.m.evidence(baseline, d.m.s.read(d.OUT / f"b1048576_s{seed}/episodes.json"), p["validation_seeds"])
        differences = [{"condition": a["condition"], "original_mean_deg": b["ppo_mean_deg"],
            "intervention_mean_deg": a["ppo_mean_deg"], "mean_delta_deg": a["ppo_mean_deg"] - b["ppo_mean_deg"],
            "original_reduction_pct": b["reduction_pct"], "intervention_reduction_pct": a["reduction_pct"],
            "original_peak_passes": sum(row["passed"] for row in b["peak_pairs"]),
            "intervention_peak_passes": sum(row["passed"] for row in a["peak_pairs"])}
            for b, a in zip(original["groups"], candidate["groups"], strict=True)]
        results[seed] = {"original": original, "intervention": candidate, "differences": differences}
        assert d.m.s.digest(path) == hashes[seed]
        print(f"Controlled gain test seed {seed}: {candidate['checks']}", flush=True)
    d.m.s.write(folder / "comparison.json", {"passed": all(result["intervention"]["passed"] for result in results.values()),
        "training_seeds": results, "all_weights_unchanged": True, "one_parameter_changed": True,
        "training": False, "final_seeds_used": False})
    for name, sha in d.m.s.read(d.OUT / "design.json")["protected_artifacts_sha256"].items():
        assert d.m.s.digest(project_file(d.m.s.ROOT, name)) == sha, name
    print("Single controlled intervention finished", flush=True)


if __name__ == "__main__":
    main()
