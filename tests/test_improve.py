from project_paths import project_file
import copy
import csv
import hashlib
import math
import unittest
from collections import defaultdict
import zipfile

import mujoco
import numpy as np

from go2_env import Go2Balance
from improve import OUT, PROTOCOL, ROOT, evidence, read


class ImprovementTests(unittest.TestCase):
    def test_deadband_preserves_pd_trajectory(self):
        baseline = Go2Balance()
        gated = Go2Balance(tilt_deadband_deg=2.0)
        baseline.reset(seed=100)
        gated.reset(seed=100)
        for _ in range(100):
            baseline.step(np.zeros(12))
            gated.step(np.ones(12))
            np.testing.assert_array_equal(baseline.data.qpos, gated.data.qpos)
        np.testing.assert_array_equal(gated.previous, np.zeros(12))

    def test_gate_blends_using_observed_tilt(self):
        env = Go2Balance(tilt_deadband_deg=2.0)
        for strength in (1.0, 0.1):
            env.strength = strength
            env.reset(seed=100)
            angle = np.deg2rad(2.5)
            env.data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
            mujoco.mj_forward(env.model, env.data)
            env.step(np.ones(12))
            np.testing.assert_allclose(env.previous, np.full(12, 0.5), atol=1e-10)

    def test_protocol_is_disjoint_and_limited(self):
        protocol = read(PROTOCOL)
        self.assertEqual(len(protocol["configurations"]), 4)
        self.assertFalse(set(protocol["validation_seeds"]) & set(protocol["final_seeds"]))
        self.assertEqual(protocol["criteria"]["required_training_seeds"], [0, 1, 2])
        self.assertEqual(protocol["criteria"]["healthy_max_extra_tilt_deg"], 0.3)
        self.assertEqual(protocol["criteria"]["severe_min_reduction"], 0.4)

    def test_extension_keeps_configuration_and_fresh_seeds(self):
        plan = read(OUT / "extension_protocol.json")
        self.assertEqual(plan["configuration_id"], "A")
        self.assertEqual(plan["training_seeds"], [0, 1, 2])
        self.assertEqual(plan["validation_seeds"], read(PROTOCOL)["validation_seeds"])
        self.assertFalse(set(plan["final_seeds"]) & set(plan["validation_seeds"] + plan["retired_test_seeds"]))
        self.assertEqual(plan["total_step_checkpoints"], [262144, 393216, 524288])

    def test_acceptance_boundaries_and_failures(self):
        protocol = read(PROTOCOL)
        seeds = [100]
        baseline = [{"seed": 100, "final_strength": strength,
                     "initial_state_sha256": "same", "simulated_s": 30.0,
                     "fallen": False, "mean_tilt_deg": 1.0} for strength in protocol["strengths"]]
        learned = copy.deepcopy(baseline)
        learned[0]["mean_tilt_deg"] = 1.3
        learned[-1]["mean_tilt_deg"] = 0.6
        self.assertTrue(evidence(baseline, learned, seeds, protocol)["passed"])
        for index, field, value in [(0, "mean_tilt_deg", 1.30001),
                                    (-1, "mean_tilt_deg", 0.60001),
                                    (2, "fallen", True), (2, "simulated_s", 29.9)]:
            failed = copy.deepcopy(learned)
            failed[index][field] = value
            self.assertFalse(evidence(baseline, failed, seeds, protocol)["passed"])
        with self.assertRaises(ValueError):
            evidence(baseline, learned[:-1], seeds, protocol)
        mismatched = copy.deepcopy(learned)
        mismatched[0]["initial_state_sha256"] = "different"
        with self.assertRaises(ValueError):
            evidence(baseline, mismatched, seeds, protocol)


class RecordedEvidenceTests(unittest.TestCase):
    def trace_means(self, path, episodes, seeds):
        groups = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.assertEqual(row["fallen"], "False")
                self.assertTrue(math.isfinite(float(row["tilt_deg"])))
                groups[int(row["seed"]), float(row["final_strength"])].append(row)
        self.assertEqual(set(groups), {(s, f) for s in seeds for f in [1, .75, .5, .25, .1]})
        for episode in episodes:
            rows = groups[episode["seed"], episode["final_strength"]]
            self.assertEqual(len(rows), 1500)
            self.assertAlmostEqual(float(rows[-1]["time_s"]), 30, places=6)
            self.assertAlmostEqual(np.mean([float(r["tilt_deg"]) for r in rows]),
                                   episode["mean_tilt_deg"], places=11)
        return {f: float(np.mean([float(r["tilt_deg"]) for (s, strength), rows in groups.items()
                                  if strength == f for r in rows])) for f in [1, .75, .5, .25, .1]}

    @unittest.skipUnless((OUT / "selection.json").exists(), "Tuning has not completed")
    def test_validation_evidence_and_selection(self):
        protocol = read(PROTOCOL)
        baseline = self.trace_means(OUT / "validation_pd_trace.csv", read(OUT / "validation_pd.json"),
                                    protocol["validation_seeds"])
        passed = []
        for name in ["A", "B", "C", "D"]:
            directory = OUT / "tuning" / name
            result = read(directory / "validation.json")
            means = self.trace_means(directory / "validation_trace.csv", result["episodes"],
                                     protocol["validation_seeds"])
            accepted = means[1] <= baseline[1] + .3 and means[.1] <= .6 * baseline[.1]
            self.assertEqual(result["passed"], accepted)
            if accepted:
                passed.append((means[.1], means[1], name))
        self.assertEqual(read(OUT / "selection.json")["selected"], min(passed)[2] if passed else None)
        self.assertFalse(read(OUT / "selection.json")["selection_uses_final_seeds"])

    @unittest.skipUnless((OUT / "final/acceptance.json").exists(), "First held-out evaluation has not completed")
    def test_first_evaluation_failure_is_preserved(self):
        protocol = read(PROTOCOL)
        final = OUT / "final"
        acceptance = read(final / "acceptance.json")
        baseline = self.trace_means(final / "pd_trace.csv", read(final / "pd.json"), protocol["final_seeds"])
        computed = []
        for seed in [0, 1, 2]:
            result = read(final / f"ppo_seed_{seed}.json")
            means = self.trace_means(final / f"ppo_seed_{seed}_trace.csv", result["episodes"], protocol["final_seeds"])
            passed = means[1] <= baseline[1] + .3 and means[.1] <= .6 * baseline[.1]
            self.assertEqual(result["passed"], passed)
            computed.append(passed)
        self.assertEqual(acceptance["passed"], all(computed))
        self.assertFalse(acceptance["passed"])

    @unittest.skipUnless((OUT / "extended/final/acceptance.json").exists(), "Extended held-out evaluation has not completed")
    def test_final_acceptance_from_raw_data(self):
        protocol = read(PROTOCOL)
        plan = read(OUT / "extension_protocol.json")
        final = OUT / "extended/final"
        acceptance = read(final / "acceptance.json")
        self.assertEqual(set(acceptance["training_seeds"]), {"0", "1", "2"})
        baseline_rows = read(final / "pd.json")
        baseline = self.trace_means(final / "pd_trace.csv", baseline_rows, plan["final_seeds"])
        initial = {(r["seed"], r["final_strength"]): r["initial_state_sha256"] for r in baseline_rows}
        for seed in [0, 1, 2]:
            result = read(final / f"ppo_seed_{seed}.json")
            means = self.trace_means(final / f"ppo_seed_{seed}_trace.csv", result["episodes"],
                                     plan["final_seeds"])
            self.assertLessEqual(means[1], baseline[1] + .3)
            self.assertLessEqual(means[.1], .6 * baseline[.1])
            self.assertTrue(result["passed"])
            for episode in result["episodes"]:
                self.assertFalse(episode["fallen"])
                self.assertAlmostEqual(episode["simulated_s"], 30, places=6)
                self.assertEqual(episode["initial_state_sha256"], initial[episode["seed"], episode["final_strength"]])
            directory = OUT / "extended" / f"round_{acceptance['total_steps']}" / f"seed_{seed}"
            config = read(directory / "config.json")
            self.assertEqual(config["training_seed"], seed)
            self.assertEqual(config["environment"], protocol["configurations"][plan["configuration_id"]])
            self.assertEqual(config["ppo"], protocol["ppo"])
            self.assertIsNone(config["cuda_build"])
            self.assertEqual(read(directory / "training.json")["steps"], acceptance["total_steps"])
            self.assertIsNone(zipfile.ZipFile(directory / "policy.zip").testzip())
        self.assertTrue(acceptance["passed"])
        design = read(OUT / "design.json")
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), design["protocol_sha256"])
        for name, checksum in design["source_sha256"].items():
            self.assertEqual(hashlib.sha256((project_file(ROOT, name)).read_bytes()).hexdigest(), checksum)
        extension = read(OUT / "extended/design.json")
        self.assertEqual(hashlib.sha256((OUT / "extension_protocol.json").read_bytes()).hexdigest(),
                         extension["plan_sha256"])
        self.assertEqual(hashlib.sha256((ROOT / "research/extend.py").read_bytes()).hexdigest(),
                         extension["runner_sha256"])

    @unittest.skipUnless((OUT / "extended/selection.json").exists(), "Extended validation has not completed")
    def test_extended_validation_precedes_final_test(self):
        plan = read(OUT / "extension_protocol.json")
        selected = read(OUT / "extended/selection.json")
        self.assertTrue(selected["passed"])
        self.assertFalse(selected["selection_uses_final_seeds"])
        self.assertIn(selected["total_steps"], plan["total_step_checkpoints"])
        baseline = self.trace_means(OUT / "validation_pd_trace.csv", read(OUT / "validation_pd.json"),
                                    plan["validation_seeds"])
        for seed in [0, 1, 2]:
            directory = OUT / "extended" / f"round_{selected['total_steps']}" / f"seed_{seed}"
            result = read(directory / "validation.json")
            means = self.trace_means(directory / "validation_trace.csv", result["episodes"],
                                     plan["validation_seeds"])
            self.assertLessEqual(means[1], baseline[1] + .3)
            self.assertLessEqual(means[.1], baseline[.1] * .6)
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
