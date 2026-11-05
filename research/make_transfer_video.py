"""Export a verified PD/student comparison using the existing serif video design."""
from project_paths import project_file

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO

import make_multimotor_video as previous
import transfer_experts as transfer

m = transfer.m
OUT = m.s.ROOT / "results/expert_transfer_video"
NAMES = ["pd", "seed_0", "seed_1", "seed_2"]
WORDS = {
    "title": "Keeping Go2 balanced",
    "subtitle": "PD compared with three students trained from the same expert library.",
    "intro": "One knee motor drops\nto 10% strength.",
    "input": "Each student uses the robot's\nusual observations.",
    "healthy": "All motors stay at full strength for this case.",
    "fault": "At 5 seconds, this motor drops from full strength to 10%.",
    "summary": "All three students passed",
    "summary_subtitle": "Final evaluation: 20 starting conditions, five cases and four controllers.",
    "scope": "Flat-floor simulation. One knee fault at 5 seconds. Hardware was not tested.",
}


def protect_results():
    p = transfer.verify()
    assert m.s.read(transfer.OUT / "completion_audit.json")["final_acceptance_passed"]
    for name, sha in m.s.read(transfer.OUT / "evidence_manifest.json").items():
        assert m.s.digest(transfer.OUT / name) == sha, name
    for name, sha in m.s.read(transfer.OUT / "design.json")["prior_artifacts_sha256"].items():
        assert m.s.digest(project_file(m.s.ROOT, name)) == sha, name
    return p


def media_helper():
    spec = importlib.util.spec_from_file_location("existing_video_media", previous.HELPER)
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    return media


def verify_plan():
    plan = m.s.read(OUT / "video_plan.json")
    assert m.s.digest(__file__) == plan["source_sha256"]
    assert m.s.digest(previous.HELPER) == plan["renderer_sha256"]
    assert m.s.digest(m.s.ROOT / "research/make_multimotor_video.py") == plan["previous_video_source_sha256"]
    assert m.s.digest(transfer.OUT / "evidence_manifest.json") == plan["result_manifest_sha256"]
    for name, item in plan["controllers"].items():
        if name != "pd":
            assert m.s.digest(m.s.ROOT / item["path"]) == item["sha256"]
    return plan


def replay(plan, config):
    motion, proof = {}, {}
    for name in NAMES:
        item = plan["controllers"][name]
        policy = None if name == "pd" else PPO.load(m.s.ROOT / item["path"], device="cpu")
        expected = {condition: [] for condition in m.CONDITIONS}
        with (transfer.OUT / f"final/{name}/trace.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["seed"]) == plan["evaluation_seed"]:
                    expected[row["condition"]].append([float(row[k]) if k != "fallen" else float(row[k] == "True") for k in m.s.FIELDS])
        summaries = {r["condition"]: r for r in m.s.read(transfer.OUT / f"final/{name}/episodes.json") if r["seed"] == plan["evaluation_seed"]}
        for condition in m.CONDITIONS:
            key = f"{condition}__{name}"
            env = m.MultiMotorBalance(motor=condition, record=True, **config)
            try:
                observation, _ = env.reset(seed=plan["evaluation_seed"])
                initial = m.s.digest_bytes(env.data.qpos.tobytes())
                assert initial == summaries[condition]["initial_state_sha256"]
                positions, tilts = [env.data.qpos.copy()], [env.metrics()["tilt_deg"]]
                for _ in range(env.horizon):
                    action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                    observation, _, terminated, _, info = env.step(action)
                    assert not terminated
                    positions.append(env.data.qpos.copy())
                    tilts.append(info["tilt_deg"])
                actual, original = np.asarray(env.samples, dtype=float), np.asarray(expected[condition])
                assert actual.shape == original.shape == (15001, 7)
                np.testing.assert_allclose(actual, original, rtol=0, atol=1e-10)
                motion[key + "_qpos"] = np.asarray(positions)
                motion[key + "_tilt"] = np.asarray(tilts)
                proof[key] = {"physics_samples": len(actual), "all_metrics_match": True,
                    "max_tilt_error_deg": float(np.abs(actual[:, 2] - original[:, 2]).max()),
                    "initial_state_sha256": initial}
            finally:
                env.close()
            print(f"Replay verified: {key}", flush=True)
    with (OUT / "motion.npz").open("xb") as handle:
        np.savez_compressed(handle, **motion)
    m.s.write(OUT / "replay_verification.json", proof)
    return motion


def intro(media, studio, motion):
    image = media.page(WORDS["title"], WORDS["subtitle"])
    media.put(image, WORDS["intro"], 80, 290, 56, width=760)
    media.put(image, WORDS["input"], 80, 490, 36, width=760)
    media.put(image, "PD + three independent students", 80, 690, 33, fill=media.SAGE, width=850)
    image.paste(studio.frame(motion["FL_calf__seed_0_qpos"][1500]).resize((924, 528)), (930, 280))
    media.put(image, "Healthy motors, then four separate knee faults", 80, 925, 34, width=1760)
    media.put(image, "Same starting pose / 2x playback / replay seed 700", 80, 1008, 27, fill=media.MUTED, width=1760)
    return image


def comparison(media, studio, motion, condition, chapter, index):
    image = Image.new("RGB", media.SIZE, "white")
    media.put(image, previous.LABELS[condition], 70, 25, 62, width=1600)
    media.put(image, WORDS["healthy"] if condition == "healthy" else WORDS["fault"], 70, 112, 30, fill=media.MUTED, width=1780)
    media.put(image, f"{chapter + 1} / 5", 1730, 45, 30, fill=media.MUTED)
    draw = ImageDraw.Draw(image)
    draw.line((960, 185, 960, 1007), fill="#c5cabf")
    draw.line((70, 605, 1850, 605), fill="#c5cabf")
    for i, name in enumerate(NAMES):
        x, y = 70 + (i % 2) * 960, 190 + (i // 2) * 425
        key = f"{condition}__{name}"
        label = "PD baseline" if i == 0 else f"Student {i} / training seed {i-1}"
        color = media.AMBER if i == 0 else media.SAGE
        media.put(image, label, x, y, 36, fill=color, width=850)
        image.paste(studio.frame(motion[key + "_qpos"][index]).resize((595, 340)), (x, y + 55))
        media.put(image, f"{motion[key + '_tilt'][index]:.2f}°", x + 705, y + 140, 65, center=True, fill=color)
        media.put(image, "Body tilt now", x + 705, y + 235, 26, center=True)
    strength = 100 if condition == "healthy" or index < 250 else 10
    media.put(image, f"Motor strength: {strength}%", 70, 1035, 26, fill=media.MUTED)
    media.put(image, "2x playback / replay seed 700", 960, 1035, 25, center=True, fill=media.MUTED)
    media.put(image, f"t = {index * .02:05.2f} s", 1695, 1035, 26, fill=media.MUTED)
    return image


def closing(media, result):
    assert result["passed"] and all(r["passed"] for r in result["training_seeds"].values())
    image = media.page(WORDS["summary"], WORDS["summary_subtitle"])
    media.put(image, "Reduction in mean tilt compared with PD", 80, 245, 38, width=1760)
    xs = [820, 1220, 1620]
    for i, x in enumerate(xs):
        media.put(image, f"Student {i+1}", x, 325, 34, center=True)
    for row, label in enumerate(["Front-left knee", "Front-right knee", "Rear-left knee", "Rear-right knee"]):
        y = 402 + row * 98
        media.put(image, label, 80, y, 38, width=600)
        for seed, x in enumerate(xs):
            value = result["training_seeds"][str(seed)]["groups"][row+1]["reduction_pct"]
            assert value >= 40
            media.put(image, f"{value:.2f}%", x, y, 42, center=True, fill=media.SAGE)
    ImageDraw.Draw(image).line((80, 810, 1840, 810), fill="#b6bdb0")
    media.put(image, "400 runs without a fall. All 240 fault peak pairs passed.", 80, 850, 35, width=1760)
    healthy = result["training_seeds"]["0"]["groups"][0]
    media.put(image, f"Healthy mean: {healthy['ppo_mean_deg']:.2f}°, the same as PD. Target: at least 40% less tilt per fault.", 80, 920, 29, width=1760)
    media.put(image, WORDS["scope"], 80, 1010, 27, fill=media.MUTED, width=1760)
    return image


def prepare():
    assert not OUT.exists(), "Refusing to overwrite a video export"
    p = protect_results()
    assert p["reserved_final_seeds"][0] == 700
    closed = m.s.read(transfer.OUT / "tuning_closed.json")
    controllers = {"pd": {"path": None}}
    for seed in range(3):
        path = transfer.OUT / f"seed_{seed}/round_3/policy.zip"
        sha = m.s.digest(path)
        assert sha == closed["checkpoint_sha256"][str(seed)]
        controllers[f"seed_{seed}"] = {"path": str(path.relative_to(m.s.ROOT)), "sha256": sha}
    OUT.mkdir()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_seed": 700,
        "selection_rule": "First seed in the completed final protocol; fixed before replay. Include all five conditions and every student. Post-evaluation illustration, not a new acceptance test.",
        "controllers": controllers, "conditions": m.CONDITIONS, "source_sha256": m.s.digest(__file__),
        "renderer_sha256": m.s.digest(previous.HELPER),
        "previous_video_source_sha256": m.s.digest(m.s.ROOT / "research/make_multimotor_video.py"),
        "result_manifest_sha256": m.s.digest(transfer.OUT / "evidence_manifest.json"),
        "words": WORDS, "design": "Existing white/serif four-panel comparison, amber PD and sage students, matched camera and framing. Natural factual English, final results separated from the illustrative replay.",
        "size": [1920, 1080], "fps": 25, "silent": True, "playback_rate": 2,
        "intro_seconds": 5, "chapter_seconds": 15, "summary_seconds": 10, "total_seconds": 90,
        "method": "Reconstruct qpos by deterministic CPU replay and match all stored 500 Hz metrics. Render copied poses. Preserve weights and all prior results. No new training or robustness experiment."}
    m.s.write(OUT / "video_plan.json", plan)
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    motion = replay(plan, p["environment"])
    media = media_helper()
    assert media.font(32).getname()[0] == "Latin Modern Roman"
    studio = media.Studio()
    studio.camera.distance = .94
    original = mujoco.MjModel.from_xml_path(str(media.MODEL))
    try:
        previous.highlight(studio, original, "FL_calf")
        intro(media, studio, motion).save(OUT / "intro.png")
        closing(media, m.s.read(transfer.OUT / "final/acceptance.json")).save(OUT / "summary.png")
        for chapter, condition in enumerate(m.CONDITIONS):
            previous.highlight(studio, original, condition)
            for index in [0, 250, 1496]:
                comparison(media, studio, motion, condition, chapter, index).save(OUT / f"preview_{chapter}_{index}.png")
    finally:
        studio.close()
    print("Replay and design previews ready", flush=True)


def render():
    plan = verify_plan()
    assert not (OUT / "go2_student_comparison.mp4").exists(), "Refusing to overwrite video"
    media = media_helper()
    motion = dict(np.load(OUT / "motion.npz", allow_pickle=False))
    studio = media.Studio()
    studio.camera.distance = .94
    original = mujoco.MjModel.from_xml_path(str(media.MODEL))
    path = OUT / "go2_student_comparison.mp4"
    writer = media.imageio_ffmpeg.write_frames(str(path), media.SIZE, fps=25, codec="libx264", quality=7,
        macro_block_size=1, ffmpeg_log_level="error", output_params=["-movflags", "+faststart", "-threads", "2"])
    writer.send(None)
    try:
        with Image.open(OUT / "intro.png") as im:
            for _ in range(125):
                writer.send(np.asarray(im.convert("RGB")))
        for chapter, condition in enumerate(m.CONDITIONS):
            previous.highlight(studio, original, condition)
            for frame in range(375):
                image = comparison(media, studio, motion, condition, chapter, frame * 4)
                writer.send(np.asarray(image))
                if frame % 125 == 0:
                    print(f"Rendered {condition}: {frame}/375", flush=True)
        with Image.open(OUT / "summary.png") as im:
            for _ in range(250):
                writer.send(np.asarray(im.convert("RGB")))
    finally:
        writer.close()
        studio.close()
    verify()


def verify():
    plan = verify_plan()
    media = media_helper()
    path = OUT / "go2_student_comparison.mp4"
    count, seconds = media.imageio_ffmpeg.count_frames_and_secs(str(path))
    assert count == 2250 and abs(seconds - 90) < .01
    reader = media.imageio_ffmpeg.read_frames(str(path))
    metadata = next(reader)
    reader.close()
    assert tuple(metadata["size"]) == (1920, 1080) and metadata["fps"] == 25
    subprocess.run([media.imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i", str(path),
                    "-f", "null", "-"], check=True, capture_output=True)
    sheet = Image.new("RGB", (1920, 1080), "white")
    times = [2, 19.96, 22.52, 34.96, 49.96, 64.96, 79.96, 83, 89]
    for i, seconds_at in enumerate(times):
        target = OUT / f"decoded_{i}.png"
        assert not target.exists(), "Refusing to overwrite decoded review frames"
        subprocess.run([media.imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(seconds_at),
                        "-i", str(path), "-frames:v", "1", str(target)], check=True, capture_output=True)
        with Image.open(target) as im:
            assert np.asarray(im).std() > 5
            sheet.paste(im.resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
    sheet.save(OUT / "contact_sheet.png")
    protect_results()
    replay_proof = m.s.read(OUT / "replay_verification.json")
    assert len(replay_proof) == 20 and all(v["all_metrics_match"] for v in replay_proof.values())
    proof = {"passed": True, "all_frames_decoded": True, "frames": count, "seconds": seconds,
        "size": metadata["size"], "fps": metadata["fps"], "silent": True, "playback_rate": 2,
        "matched_physics_samples": sum(v["physics_samples"] for v in replay_proof.values()),
        "max_replay_tilt_error_deg": max(v["max_tilt_error_deg"] for v in replay_proof.values()),
        "existing_results_and_checkpoints_unchanged": True, "video_sha256": m.s.digest(path),
        "motion_sha256": m.s.digest(OUT / "motion.npz"), "font_sha256": m.s.digest(media.FONT_DIR / "lmroman10-regular.otf")}
    m.s.write(OUT / "verification.json", proof)
    with (OUT / "README.md").open("x", encoding="utf-8") as handle:
        handle.write("# Go2 student comparison\n\n90-second silent 1080p/25fps film: a five-second introduction, five 15-second cases at 2x playback, and a ten-second final-evaluation summary. Text is English; layout reuses the existing white/serif comparison design.\n\n"
            "PD and the three final students use evaluation seed 700, the first seed in the completed final protocol, fixed before replay. All five cases and every student are included. No motor identity is passed to student inference. Motion is reconstructed by deterministic replay because the stored traces do not contain full joint poses; all 300,020 physics samples match the original measurements. Summary percentages are from all twenty final starts per condition, not only the displayed run. Student labels 1-3 correspond to training seeds 0-2.\n\n"
            "Existing checkpoints and all protected experiment artifacts retain their hashes. No training, new robustness test, narration, music or CUDA was used. Display styling changes only a copied rendering model. Scope remains a flat floor, one knee weakened to 10% at 5 seconds; physical hardware was not tested.\n\n"
            "See video_plan.json, replay_verification.json, verification.json and contact_sheet.png for provenance and checks.\n")
    print(f"Verified film ready: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "render"])
    {"prepare": prepare, "render": render}[parser.parse_args().stage]()
