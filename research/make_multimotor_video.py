"""Replay one fixed evaluation seed, verify its physics traces, and export a comparison film."""

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

import multimotor as m

OUT = m.s.ROOT / "results/multimotor_video"
HELPER = Path(r"C:\Users\Zidan\Documents\Codex\work\balance-media\make_media.py")
LABELS = {"healthy": "Healthy motors", "FL_calf": "Front-left knee motor",
          "FR_calf": "Front-right knee motor", "RL_calf": "Rear-left knee motor",
          "RR_calf": "Rear-right knee motor"}


def protect_results():
    for name, checksum in m.s.read(m.OUT / "evidence_manifest.json").items():
        assert m.s.digest(m.OUT / name) == checksum, name
    m.verify()


def replay(plan, paths, config):
    arrays, proof = {}, {}
    for name, path in paths.items():
        policy = None if path is None else PPO.load(path, device="cpu")
        with (m.OUT / f"final/{name}/trace.csv").open(encoding="utf-8") as handle:
            expected = {condition: [] for condition in m.CONDITIONS}
            for row in csv.DictReader(handle):
                if int(row["seed"]) == plan["evaluation_seed"]:
                    expected[row["condition"]].append([float(row[k]) if k != "fallen" else float(row[k] == "True")
                                                       for k in m.s.FIELDS])
        summaries = {r["condition"]: r for r in m.s.read(m.OUT / f"final/{name}/episodes.json")
                     if r["seed"] == plan["evaluation_seed"]}
        for condition in m.CONDITIONS:
            key = f"{condition}__{name}"
            env = m.MultiMotorBalance(motor=condition, record=True, **({} if path is None else config))
            observation, _ = env.reset(seed=plan["evaluation_seed"])
            initial_hash = m.s.digest_bytes(env.data.qpos.tobytes())
            assert initial_hash == summaries[condition]["initial_state_sha256"]
            positions, tilts = [env.data.qpos.copy()], [env.metrics()["tilt_deg"]]
            for _ in range(1500):
                action = np.zeros(12) if policy is None else policy.predict(observation, deterministic=True)[0]
                observation, _, terminated, _, info = env.step(action)
                assert not terminated
                positions.append(env.data.qpos.copy())
                tilts.append(info["tilt_deg"])
            actual, original = np.asarray(env.samples, dtype=float), np.asarray(expected[condition])
            assert actual.shape == original.shape == (15001, 7)
            np.testing.assert_allclose(actual, original, rtol=0, atol=1e-10)
            arrays[key + "_qpos"] = np.array(positions)
            arrays[key + "_tilt"] = np.array(tilts)
            proof[key] = {"physics_samples": len(actual), "all_recorded_metrics_match": True,
                          "max_tilt_error_deg": float(np.max(np.abs(actual[:, 2] - original[:, 2]))),
                          "initial_state_sha256": initial_hash}
            env.close()
            print(f"Replay verified: {key}", flush=True)
    np.savez_compressed(OUT / "motion.npz", **arrays)
    m.s.write(OUT / "replay_verification.json", proof)
    return arrays


def highlight(studio, original, condition):
    for name in m.MOTORS:
        body = studio.model.body(name).id
        mask = (studio.model.geom_bodyid == body) & (studio.model.geom_group == 2)
        studio.model.geom_matid[mask] = original.geom_matid[mask]
        studio.model.geom_rgba[mask] = original.geom_rgba[mask]
        if name == condition:
            assert mask.any()
            studio.model.geom_matid[mask] = -1
            studio.model.geom_rgba[mask] = [.83, .58, .27, 1]


def comparison(media, studio, motion, condition, chapter, index):
    image = Image.new("RGB", media.SIZE, "white")
    media.put(image, LABELS[condition], 70, 25, 62, width=1600)
    subtitle = ("All motors remain at full strength throughout this run." if condition == "healthy" else
                f"The {LABELS[condition].lower()} drops to 10% strength at 5 seconds.")
    media.put(image, subtitle, 70, 112, 30, fill=media.MUTED, width=1780)
    media.put(image, f"{chapter + 1} / 5", 1730, 45, 30, fill=media.MUTED)
    draw = ImageDraw.Draw(image)
    draw.line((960, 185, 960, 1010), fill="#c5cabf")
    draw.line((70, 605, 1850, 605), fill="#c5cabf")
    for i, name in enumerate(["pd", "ppo_seed_0", "ppo_seed_1", "ppo_seed_2"]):
        x, y = 70 + (i % 2) * 960, 190 + (i // 2) * 425
        key = f"{condition}__{name}"
        label = "Fixed joint targets" if i == 0 else f"Learned adjustments / model {i}"
        color = media.AMBER if i == 0 else media.SAGE
        media.put(image, label, x, y, 36, fill=color, width=850)
        image.paste(studio.frame(motion[key + "_qpos"][index]).resize((595, 340)), (x, y + 55))
        media.put(image, f"{motion[key + '_tilt'][index]:.2f}°", x + 705, y + 140, 65, center=True, fill=color)
        media.put(image, "Body tilt now", x + 705, y + 235, 26, center=True)
    strength = 100 if condition == "healthy" or index < 250 else 10
    media.put(image, f"Motor strength: {strength}%", 70, 1035, 26, fill=media.MUTED)
    media.put(image, "2× playback / same starting pose", 960, 1035, 25, center=True, fill=media.MUTED)
    media.put(image, f"t = {index * .02:05.2f} s", 1695, 1035, 26, fill=media.MUTED)
    return image


def closing(media, result):
    assert not result["passed"]
    assert all(r["checks"]["survival"] and r["checks"]["healthy"] for r in result["training_seeds"].values())
    image = media.page("The full test: targets not met", "Summary of all 20 starting conditions, not just the run shown here.")
    lines = ["All 400 runs completed 30 seconds without a fall.",
             "With healthy motors, both approaches averaged 1.19° of body tilt.",
             "Model 3 reduced tilt by 49.83% for the front-right motor.",
             "The other motor and model combinations missed the 40% target.",
             "Model 2 exceeded the reference peak in all 20 rear-right runs."]
    for i, line in enumerate(lines):
        media.put(image, line, 90, 270 + 125 * i, 39, width=1740)
    media.put(image, "Models 1, 2 and 3 were trained with different random seeds.",
              960, 970, 29, center=True, fill=media.MUTED)
    media.put(image, "Video: one matched starting condition across all five cases.",
              960, 1030, 27, center=True, fill=media.MUTED)
    return image


def main():
    protect_results()
    assert not OUT.exists(), "Refusing to overwrite an existing video export"
    protocol = m.verify()
    seed = protocol["final_seeds"][0]
    assert seed == 600
    selected = m.s.read(m.OUT / "selection.json")["selected"]
    paths = {"pd": None, "ppo_seed_0": m.OUT / f"candidates/{selected}/policy.zip",
             "ppo_seed_1": m.OUT / "replicates/seed_1/policy.zip",
             "ppo_seed_2": m.OUT / "replicates/seed_2/policy.zip"}
    hashes = {name: m.s.digest(path) for name, path in paths.items() if path is not None}
    assert hashes == {f"ppo_seed_{key}": sha for key, sha in m.s.read(m.OUT / "tuning_closed.json")["checkpoint_sha256"].items()}
    OUT.mkdir()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_seed": seed,
            "selection_rule": "First seed listed in the completed final-evaluation protocol. Fixed before replay; all five conditions and all three models are included. This video is a post-evaluation illustration, not a preregistered acceptance test.",
            "conditions": m.CONDITIONS, "checkpoint_sha256": hashes,
            "source_sha256": m.s.digest(__file__), "renderer_sha256": m.s.digest(HELPER),
            "result_manifest_sha256": m.s.digest(m.OUT / "evidence_manifest.json"),
            "playback_rate": 2, "simulated_seconds_per_case": 30, "video_seconds_per_case": 15,
            "summary_seconds": 5, "total_seconds": 80, "fps": 25, "size": [1920, 1080], "silent": True,
            "method": "Deterministic replay matched to every stored physics sample for seed 600, rendered from copied poses. The source traces do not store joint poses, so these are reconstructed trajectories, not previously recorded video. Video subsamples control poses; peak statistics use the original 500 Hz data."}
    m.s.write(OUT / "video_plan.json", plan)
    torch.set_num_threads(1)
    assert torch.version.cuda is None
    motion = replay(plan, paths, protocol["configurations"][selected])
    spec = importlib.util.spec_from_file_location("previous_media", HELPER)
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    assert media.font(32).getname()[0] == "Latin Modern Roman"
    result = m.s.read(m.OUT / "final/acceptance.json")
    end = closing(media, result)
    end.save(OUT / "summary.png")
    original = mujoco.MjModel.from_xml_path(str(media.MODEL))
    studio = media.Studio()
    dest = OUT / "balance_across_four_motors.mp4"
    writer = media.imageio_ffmpeg.write_frames(str(dest), media.SIZE, fps=25, codec="libx264", quality=7,
              macro_block_size=1, ffmpeg_log_level="error",
              output_params=["-movflags", "+faststart", "-threads", "2"])
    writer.send(None)
    try:
        for chapter, condition in enumerate(m.CONDITIONS):
            highlight(studio, original, condition)
            for frame in range(375):
                image = comparison(media, studio, motion, condition, chapter, frame * 4)
                writer.send(np.asarray(image))
                if frame in [0, 75, 374]:
                    image.save(OUT / f"{chapter}_frame_{frame:03}.png")
                if frame % 125 == 0:
                    print(f"Rendered {condition}: {frame}/375 frames", flush=True)
        for _ in range(125):
            writer.send(np.asarray(end))
    finally:
        writer.close()
        studio.close()
    frames, seconds = media.imageio_ffmpeg.count_frames_and_secs(str(dest))
    assert frames == 2000 and abs(seconds - 80) < .01
    reader = media.imageio_ffmpeg.read_frames(str(dest))
    metadata = next(reader)
    reader.close()
    assert tuple(metadata["size"]) == (1920, 1080) and metadata["fps"] == 25
    sheet = Image.new("RGB", (1920, 720), "white")
    for i, seconds in enumerate([14.96, 29.96, 44.96, 59.96, 74.96, 78]):
        frame_path = OUT / f"decoded_{i}.png"
        subprocess.run([media.imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(seconds),
                        "-i", str(dest), "-frames:v", "1", "-y", str(frame_path)], check=True, capture_output=True)
        with Image.open(frame_path) as image:
            assert np.asarray(image).std() > 5
            sheet.paste(image.resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
    sheet.save(OUT / "contact_sheet.png")
    protect_results()
    assert all(m.s.digest(paths[name]) == sha for name, sha in hashes.items())
    assert m.s.digest(HELPER) == plan["renderer_sha256"]
    m.s.write(OUT / "verification.json", {
        "passed": True, "seed": seed, "reconstructed_replay": True,
        "matched_physics_samples": 20 * 15001, "all_metrics_match_original_traces": True,
        "all_conditions_and_models_included": True, "checkpoints_and_existing_results_unchanged": True,
        "decoded_frames": frames, "all_frames_decoded": True, "seconds": 80, "size": metadata["size"],
        "fps": metadata["fps"], "playback_rate": 2, "silent": True,
        "font_file": str(media.FONT_DIR / "lmroman10-regular.otf"),
        "font_sha256": m.s.digest(media.FONT_DIR / "lmroman10-regular.otf"),
        "video_sha256": m.s.digest(dest), "motion_sha256": m.s.digest(OUT / "motion.npz")})
    (OUT / "README.md").write_text("# Balance across four knee-motor faults\n\n"
        "One silent 80-second video: five 30-second simulated cases at 2× playback, then a five-second summary. "
        "PD and all three selected models share the same starting condition, evaluation seed 600. "
        "This seed was selected as the first seed in the existing protocol before replay, after the experiment had finished. "
        "It was not chosen as the best-looking run.\n\n"
        "Motion comes from deterministic simulator replay, checked against all 300,020 stored physics samples for this seed. "
        "Joint poses are reconstructed because the original traces contain measured tilt and height rather than full joint poses. "
        "The whole evaluation still fails its criteria; the final summary includes the failed peak comparison. "
        "Model labels 1, 2 and 3 refer to training seeds 0, 1 and 2. "
        "Colors and the viewing angle affect presentation only; existing checkpoints and experiment files are unchanged.\n\n"
        "See `video_plan.json`, `replay_verification.json`, `verification.json`, and `contact_sheet.png` for provenance and checks.\n",
        encoding="utf-8")
    print(f"Verified video ready: {dest}", flush=True)


if __name__ == "__main__":
    main()
