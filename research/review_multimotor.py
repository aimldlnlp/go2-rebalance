"""Independently validate saved experiment data and export the scientific readout."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import multimotor as m


def verify_dataset(folder, seeds):
    raw = pd.read_csv(folder / "trace.csv", engine="python")
    episodes = m.s.read(folder / "episodes.json")
    expected = {(condition, seed) for condition in m.CONDITIONS for seed in seeds}
    assert len(episodes) == len(expected)
    assert set(zip(raw.condition, raw.seed)) == expected
    assert {(r["condition"], r["seed"]) for r in episodes} == expected
    for row in episodes:
        group = raw.loc[(raw.condition == row["condition"]) & (raw.seed == row["seed"])]
        assert len(group) == row["physics_samples"]
        np.testing.assert_array_equal(group.tick, np.arange(len(group)))
        np.testing.assert_allclose(group.time_s, group.tick * .002, rtol=0, atol=1e-12)
        strength = 1 if row["condition"] == "healthy" else .1
        np.testing.assert_array_equal(group.strength_applied, np.where(group.tick <= 2500, 1, strength))
        np.testing.assert_array_equal(group.strength_next_step, np.where(group.tick < 2500, 1, strength))
        assert np.isfinite(group[["tilt_deg", "height_m", "time_s"]].to_numpy()).all()
        post = group.loc[group.tick > 2500]
        mean_post = math.fsum(post.tilt_deg) / len(post)
        whole = group.loc[group.tick > 0]
        mean_whole = math.fsum(whole.tilt_deg) / len(whole)
        np.testing.assert_allclose(mean_post, row["mean_post_tilt_deg"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(mean_whole, row["mean_tilt_deg"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(group.tilt_deg.max(), row["max_tilt_deg"], rtol=0, atol=1e-12)
        assert bool(group.fallen.any()) == row["fallen"]
        assert bool(((group.height_m < .16) | (group.tilt_deg > 45)).any()) == row["fallen"]
        assert row["simulated_s"] == (len(group) - 1) * .002
        reference = group.loc[(group.tick >= 2000) & (group.tick <= 2500)].tilt_deg.mean()
        good = post.tilt_deg.to_numpy() <= reference + .5
        recovery = None
        if good.all():
            recovery = 0.0
        elif len(good) >= 501:
            cs = np.r_[0, np.cumsum(good)]
            windows = np.flatnonzero(cs[501:] - cs[:-501] == 501)
            if len(windows):
                recovery = float((post.tick.iloc[windows[0]] - 2500) * .002)
        assert recovery == row["recovery_s"]
    return episodes, len(raw)


def independent_comparison(baseline, learned, seeds):
    before = {(r["condition"], r["seed"]): r for r in baseline}
    after = {(r["condition"], r["seed"]): r for r in learned}
    assert before.keys() == after.keys()
    checks = {"survival": all(not r["fallen"] and r["simulated_s"] == 30 for r in baseline + learned)}
    groups = []
    for condition in m.CONDITIONS:
        pairs = [(before[condition, seed], after[condition, seed]) for seed in seeds]
        assert all(b["initial_state_sha256"] == a["initial_state_sha256"] for b, a in pairs)
        field = "mean_tilt_deg" if condition == "healthy" else "mean_post_tilt_deg"
        bmean = math.fsum(b[field] for b, _ in pairs) / len(seeds)
        amean = math.fsum(a[field] for _, a in pairs) / len(seeds)
        peaks = sum(a["max_tilt_deg"] <= b["max_tilt_deg"] for b, a in pairs)
        if condition == "healthy":
            checks["healthy"] = amean <= bmean + .3
        else:
            checks[condition + "_mean"] = amean <= .6 * bmean
            checks[condition + "_peak"] = peaks == len(seeds)
        groups.append({"condition": condition, "pd_mean_deg": bmean, "ppo_mean_deg": amean,
                       "reduction_pct": 100 * (1 - amean / bmean), "delta_deg": amean - bmean,
                       "peak_passes": peaks, "episodes": len(seeds),
                       "peak_pairs": [{"seed": b["seed"], "pd_peak_deg": b["max_tilt_deg"],
                                       "ppo_peak_deg": a["max_tilt_deg"],
                                       "passed": a["max_tilt_deg"] <= b["max_tilt_deg"]} for b, a in pairs],
                       "survived": sum(not a["fallen"] and a["simulated_s"] == 30 for _, a in pairs),
                       "recovered": sum(a["recovered"] for _, a in pairs)})
    return {"passed": all(checks.values()), "checks": checks, "groups": groups}


def assert_comparison(independent, recorded):
    assert independent["passed"] == recorded["passed"]
    assert independent["checks"] == recorded["checks"]
    for a, b in zip(independent["groups"], recorded["groups"], strict=True):
        assert a["condition"] == b["condition"]
        for field in ["pd_mean_deg", "ppo_mean_deg", "reduction_pct", "delta_deg"]:
            np.testing.assert_allclose(a[field], b[field], rtol=0, atol=1e-10)
        assert a["peak_pairs"] == b["peak_pairs"]


def main():
    p = m.verify()
    design = m.s.read(m.OUT / "design.json")
    for path, sha in design["prior_artifacts_sha256"].items():
        assert m.s.digest(m.s.ROOT / path) == sha, path
    assert "Ran 28 tests" in (m.OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    assert "\nOK" in (m.OUT / "tests-before.log").read_text(encoding="utf-8-sig")
    selection = m.s.read(m.OUT / "selection.json")
    selected = selection["selected"]
    close = m.s.read(m.OUT / "tuning_closed.json")
    execution = m.s.read(m.OUT / "execution.json")
    assert datetime.fromisoformat(p["created_utc"]) < datetime.fromisoformat(execution["started_utc"])
    assert datetime.fromisoformat(execution["started_utc"]) < datetime.fromisoformat(close["created_utc"])
    for path in (m.OUT / "final").rglob("trace.csv"):
        stat = path.stat()
        created = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime), timezone.utc)
        assert datetime.fromisoformat(close["created_utc"]) <= created
    assert execution["cuda_build"] is None and not execution["rendering"]
    datasets = {}
    val_pd, count = verify_dataset(m.OUT / "validation/pd", p["validation_seeds"])
    datasets["validation/pd"] = count
    candidates = {}
    for key in p["configurations"]:
        folder = m.OUT / "candidates" / key
        episodes, count = verify_dataset(folder / "validation", p["validation_seeds"])
        datasets[f"candidates/{key}/validation"] = count
        candidates[key] = independent_comparison(val_pd, episodes, p["validation_seeds"])
        assert_comparison(candidates[key], m.s.read(folder / "validation.json"))
        metadata = m.s.read(folder / "training.json")
        assert metadata["actual_steps"] == 262144
        assert m.s.digest(folder / "policy.zip") == metadata["checkpoint_sha256"]
    assert sorted(path.name for path in (m.OUT / "candidates").iterdir()) == list("ABCD")
    def rank(key):
        c = candidates[key]
        return (sum(not v for v in c["checks"].values()),
                max(g["ppo_mean_deg"] / g["pd_mean_deg"] for g in c["groups"][1:]),
                sum(g["episodes"] - g["peak_passes"] for g in c["groups"][1:]), c["groups"][0]["delta_deg"], key)
    assert selected == min(candidates, key=rank)
    assert not selection["selection_uses_final_seeds"]
    pilot, count = verify_dataset(m.OUT / "candidates/A/pilot", [100])
    datasets["candidates/A/pilot"] = count
    baseline, count = verify_dataset(m.OUT / "final/pd", p["final_seeds"])
    datasets["final/pd"] = count
    results = {}
    recorded = m.s.read(m.OUT / "final/acceptance.json")
    coverage, training = {}, {}
    for seed in [0, 1, 2]:
        folder = m.OUT / "candidates" / selected if seed == 0 else m.OUT / "replicates" / f"seed_{seed}"
        meta = m.s.read(folder / "training.json")
        training[str(seed)] = meta
        assert meta["actual_steps"] == 262144
        assert m.s.digest(folder / "policy.zip") == close["checkpoint_sha256"][str(seed)] == meta["checkpoint_sha256"]
        monitor = pd.read_csv(folder / "monitor.csv", comment="#")
        coverage[str(seed)] = {str(k): int(v) for k, v in monitor.fault_motor.value_counts().items()}
        assert set(coverage[str(seed)]) == set(m.CONDITIONS)
        assert monitor.final_strength.between(.1, 1).all() and monitor.fault_onset_s.between(3, 8).all()
        if seed:
            val, count = verify_dataset(folder / "validation", p["validation_seeds"])
            datasets[f"replicates/seed_{seed}/validation"] = count
            assert_comparison(independent_comparison(val_pd, val, p["validation_seeds"]), m.s.read(folder / "validation.json"))
        episodes, count = verify_dataset(m.OUT / f"final/ppo_seed_{seed}", p["final_seeds"])
        datasets[f"final/ppo_seed_{seed}"] = count
        results[str(seed)] = independent_comparison(baseline, episodes, p["final_seeds"])
        assert_comparison(results[str(seed)], recorded["training_seeds"][str(seed)])
    assert recorded["passed"] == all(r["passed"] for r in results.values())
    font_path = Path(r"C:\Users\Zidan\AppData\Local\Programs\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, (fault_ax, healthy_ax) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2.4, 1]})
    x, width = np.arange(4), .24
    colors = ["#355c7d", "#62815c", "#9a6544"]
    for seed in range(3):
        values = [g["reduction_pct"] for g in results[str(seed)]["groups"][1:]]
        bars = fault_ax.bar(x + (seed - 1) * width, values, width, label=f"Model {seed + 1}", color=colors[seed])
        fault_ax.bar_label(bars, fmt="%.1f", fontsize=10, padding=4)
    fault_ax.axhline(40, color="#222222", linestyle="--", linewidth=1, label="Required reduction: 40%")
    fault_ax.axhline(0, color="#666666", linewidth=.6)
    fault_ax.set_xticks(x, ["Front left", "Front right", "Rear left", "Rear right"])
    fault_ax.set_ylabel("Reduction in mean body tilt (%)")
    fault_ax.set_title("At 10% knee-motor strength")
    fault_ax.legend(loc="best", fontsize=10)
    values = [results[str(seed)]["groups"][0]["delta_deg"] for seed in range(3)]
    bars = healthy_ax.bar(range(3), values, color=colors)
    healthy_ax.bar_label(bars, fmt="%.3f", padding=4)
    healthy_ax.axhline(.3, color="#222222", linestyle="--", linewidth=1)
    healthy_ax.set_ylim(min(-.05, min(values) - .05), max(.4, max(values) + .1))
    healthy_ax.set_xticks(range(3), ["Model 1", "Model 2", "Model 3"])
    healthy_ax.set_ylabel("Mean tilt above fixed targets (°)")
    healthy_ax.set_title("Healthy motors: limit +0.3°")
    fig.suptitle("Balance across four knee-motor faults", fontsize=21)
    fig.text(.5, .02, "Twenty matched runs per condition and model. Negative reduction means more tilt. Peaks and falls are assessed separately.", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(m.OUT / "comparison.png", dpi=160)
    plt.close(fig)
    rows = []
    for seed, r in results.items():
        for g in r["groups"]:
            rows.append(f"| {seed} | {g['condition']} | {g['pd_mean_deg']:.4f} | {g['ppo_mean_deg']:.4f} | {g['reduction_pct']:.2f}% | {g['survived']}/20 | {'N/A' if g['condition'] == 'healthy' else str(g['peak_passes']) + '/20'} |")
    candidate_table = [f"| {key} | {r['passed']} | {sum(not v for v in r['checks'].values())} | {max(g['ppo_mean_deg']/g['pd_mean_deg'] for g in r['groups'][1:]):.4f} |" for key, r in candidates.items()]
    report = f"""# Learning balance across four knee motors

**Held-out acceptance: {'PASS' if recorded['passed'] else 'FAIL'}.** Configuration {selected} was selected using only validation seeds 100–104. It was trained from scratch with three independent training seeds at 262,144 steps each, then evaluated on the previously unused seeds 600–619. No criteria were relaxed after seeing results.

![Comparison across motors](comparison.png)

| Training seed | Condition | PD mean tilt (°) | Learned mean tilt (°) | Reduction vs PD | Survived 30 s | Paired peak checks |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Healthy means cover (0,30] seconds. Fault means cover (5,30] seconds; these distinct windows match their respective frozen criteria. Reduction percentages average the episode means first, then compare against matched PD means. Negative reduction means the learned controller tilted more. Healthy acceptance uses an absolute margin of 0.3°, not a percentage. Any fall or early termination fails survival. Each faulted motor independently requires at least 40% reduction and no larger peak in any paired episode. The chart shows means only; it cannot establish peak or survival acceptance. Per-checkpoint checks and every peak pair are retained in `final/acceptance.json`.

Chart models 1, 2 and 3 correspond to training seeds 0, 1 and 2. Independent checkpoint acceptance: {', '.join('seed ' + seed + ': ' + ('PASS' if r['passed'] else 'FAIL') for seed, r in results.items())}.

## Candidate selection and the pilot

| Configuration | Validation accepted | Failed validation checks | Worst motor PPO/PD mean ratio |
|---|---|---:|---:|
{chr(10).join(candidate_table)}

Exactly four configurations were trained with seed 0. They combine residual limits of 0.25 or 0.35 rad with tilt penalties of 5 or 15; other settings are fixed. Configuration A's 16,384-step pilot was part of its 262,144-step budget, not an extra candidate. Pilot parameters were finite, saving/reloading preserved deterministic predictions, and all five conditions were exercised before continuing. Pilot outcomes remain in `candidates/A/pilot`.

The preregistered ranking prioritizes the fewest failed validation checks, then the worst motor mean ratio, total failed peak pairs, healthy mean excess, and configuration ID. Selection validity does not imply performance acceptance: if all candidates fail, the best candidate still receives the preregistered three-seed evaluation and remains a failure unless it actually meets the unchanged criteria. Validation acceptance for the selected candidate was {selection['validation_passed']}.

## Training and evaluation

Each training reset uniformly chooses healthy operation or one of the four knee motors. Fault strength is 10%, 25%, 50%, or 75%, with probabilities 0.5, 0.2, 0.2, and 0.1. Fault onset is sampled between 3 and 8 seconds. The policy receives the original 60 observations and does not receive motor identity, strength or onset. It applies 12 bounded corrections to joint targets through the original PD controller. The 2° deadband and one-degree blend remain. No action mirroring, pretrained initialization, physical-model edits, or changes to PD gains were used.

Evaluation uses healthy operation and four separate 100%-to-10% knee-motor changes at 5 seconds. Initial noise is ±0.01 rad; each controller sees the same initial state for a seed. All five conditions are run for every seed. The physical timestep is 0.002 seconds; the policy interval is 0.02 seconds. Peak tilt is measured at every physics step over the full observed episode, including the healthy phase. The 5-second sample reflects the preceding full-strength interval; the first affected integration sample is at 5.002 seconds. Recovery uses the previous [4,5]-second tilt reference plus 0.5° held for one second; null recovery times remain null, not zero.

The protocol was written at {p['created_utc']}; execution began at {execution['started_utc']}. Tuning closed at {close['created_utc']}, before the final evaluation files were created. Old validation seeds were reused deliberately. Final seeds are disjoint from all recorded earlier training/evaluation seed sets. This is a new held-out assessment of four trained fault categories, not evidence about untrained failure types.

## Evidence review using Data

Ready within reviewed scope: raw traces were checked for exact condition/seed coverage, duplicate or missing ticks, strength boundaries, finite values, fall flags, durations, and the distinct healthy/fault measurement windows. Means were independently recomputed with `math.fsum`, peaks from raw columns, and recovery from cumulative one-second windows. Acceptance, validation ranking, final checkpoint hashes, and the split between selection and final testing were independently checked. Agreement between two calls to the same acceptance helper was not used as independent proof.

All 28 functional/regression tests passed before training. Training monitor logs confirm every selected model encountered healthy operation and all four motor faults. The three selected checkpoints and all {len(design['prior_artifacts_sha256'])} protected prior artifacts retain their saved hashes. No CUDA or rendering was used. `data_validation.json` records dataset row counts, training-condition coverage, and the reviewed checks; `evidence_manifest.json` provides artifact hashes. The chart is a static scientific figure for this repository's existing Markdown report, not an interactive dashboard.

Checkpoint paths:

- Seed 0: `candidates/{selected}/policy.zip`
- Seed 1: `replicates/seed_1/policy.zip`
- Seed 2: `replicates/seed_2/policy.zip`

`protocol.json`, `design.json`, `selection.json`, `tuning_closed.json`, candidate/replicate configuration and training files, raw `trace.csv` files, `training.log`, and `tests-before.log` retain the inspectable inputs and calculations. Candidate failures remain available. To recheck the result, run `.venv\\Scripts\\python.exe review_multimotor.py` from the project directory. Training commands intentionally refuse overwriting this experiment.

## Limits

Three training seeds and twenty initial-condition seeds per condition do not establish broad reliability or confidence bounds for all disturbances. The bars show separate models, not confidence intervals. Training strengths and onset times were randomized, but the final fault evaluation fixes 10% strength and a 5-second onset. Small initial-position noise, one affected knee at a time, and a flat simulated floor limit the scope. Walking, simultaneous failures, other joint types, rough terrain, hardware transfer, and longer-term stability were not tested. Sampled peaks do not bound continuous motion between physics ticks. This fixed-budget experiment makes no claim about how much more training would be required to pass.
"""
    (m.OUT / "REPORT.md").write_text(report, encoding="utf-8")
    validation = {"completed": True, "review_status": "Ready within reviewed scope", "acceptance_passed": recorded["passed"],
                  "independent_raw_means_peaks_recovery": True, "validation_only_selection_verified": True,
                  "all_three_checkpoints_verified": True, "prior_artifacts_unchanged": len(design["prior_artifacts_sha256"]),
                  "training_condition_coverage": coverage, "datasets_raw_rows": datasets, "tests_passed_before_training": 28,
                  "training_metadata": training, "review_source_sha256": m.s.digest(__file__)}
    (m.OUT / "data_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    manifest = {str(path.relative_to(m.OUT)): m.s.digest(path) for path in sorted(m.OUT.rglob("*"))
                if path.is_file() and path.name != "evidence_manifest.json"}
    (m.OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Independent Data-method review complete; accepted:", recorded["passed"])
    print("\n".join(rows))


if __name__ == "__main__":
    main()
