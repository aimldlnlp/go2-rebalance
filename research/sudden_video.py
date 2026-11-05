"""Render the preregistered seed directly from the evaluation's recorded poses."""

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import sudden as s

HELPER = Path(r"C:\Users\Zidan\Documents\Codex\work\balance-media\make_media.py")


def main():
    protocol = s.verify_frozen()
    assert s.digest(HELPER) == protocol["video_renderer_sha256"]
    spec = importlib.util.spec_from_file_location("previous_media", HELPER)
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    assert media.font(32).getname()[0] == "Latin Modern Roman"
    motion = np.load(s.OUT / "video_motion.npz")
    names = protocol["video_controllers"]
    for name in names:
        with (s.OUT / f"{name}_trace.csv").open() as handle:
            import csv
            raw = [float(r["tilt_deg"]) for r in csv.DictReader(handle)
                   if int(r["seed"]) == protocol["video_seed"] and int(r["tick"]) % 10 == 0]
        np.testing.assert_array_equal(raw, motion[name + "_tilt"])
    dest = s.OUT / "sudden_comparison_seed_400.mp4"
    assert not dest.exists(), "Refusing to replace the selected example"
    studio = media.Studio()
    writer = media.imageio_ffmpeg.write_frames(str(dest), media.SIZE, fps=25, codec="libx264", quality=7,
                 macro_block_size=1, ffmpeg_log_level="error",
                 output_params=["-movflags", "+faststart", "-threads", "2"])
    writer.send(None)
    try:
        for frame in range(750):
            index = frame * 2
            seconds = index * .02
            image = Image.new("RGB", media.SIZE, "white")
            media.put(image, "Balance after a sudden loss of strength", 70, 25, 62, width=1780)
            media.put(image, "One front-left knee motor drops from full strength to 10% at 5 seconds.",
                      70, 112, 30, fill=media.MUTED, width=1780)
            draw = ImageDraw.Draw(image)
            draw.line((960, 185, 960, 1010), fill="#c5cabf")
            draw.line((70, 605, 1850, 605), fill="#c5cabf")
            for i, name in enumerate(names):
                x, y = 70 + (i % 2) * 960, 190 + (i // 2) * 425
                label = "Fixed joint targets" if i == 0 else f"Learned adjustments / model {i}"
                color = media.AMBER if i == 0 else media.SAGE
                media.put(image, label, x, y, 36, fill=color, width=850)
                positions = motion[name + "_qpos"]
                pose_index = min(index, len(positions) - 1)
                image.paste(studio.frame(positions[pose_index]).resize((595, 340)), (x, y + 55))
                media.put(image, f"{motion[name + '_tilt'][pose_index]:.2f}°", x + 705, y + 140,
                          65, center=True, fill=color)
                media.put(image, "Body tilt now", x + 705, y + 235, 26, center=True)
                if index >= len(positions):
                    media.put(image, "Run ended", x + 705, y + 285, 25, center=True, fill=media.AMBER)
            strength = 100 if seconds < 5 else 10
            media.put(image, f"Knee motor strength: {strength}%", 70, 1035, 26, fill=media.MUTED)
            media.put(image, "Matched starting pose / evaluation seed 400", 960, 1035, 25, center=True,
                      fill=media.MUTED)
            media.put(image, f"{seconds:04.2f} s", 1720, 1035, 26, fill=media.MUTED)
            writer.send(np.asarray(image))
            if frame in [0, 124, 125, 150, 375, 749]:
                image.save(s.OUT / f"video_frame_{frame:03}.png")
            if frame % 125 == 0:
                print(f"Video frame {frame}/750", flush=True)
    finally:
        writer.close()
        studio.close()
    frames, seconds = media.imageio_ffmpeg.count_frames_and_secs(str(dest))
    assert frames == 750 and abs(seconds - 30) < .01
    reader = media.imageio_ffmpeg.read_frames(str(dest))
    metadata = next(reader)
    reader.close()
    assert tuple(metadata["size"]) == (1920, 1080) and metadata["fps"] == 25
    sheet = Image.new("RGB", (1920, 720), "white")
    for i, t in enumerate([0, 4.96, 5, 6, 15, 29.96]):
        frame_path = s.OUT / f"decoded_frame_{i}.png"
        subprocess.run([media.imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(t),
                        "-i", str(dest), "-frames:v", "1", "-y", str(frame_path)],
                       check=True, capture_output=True)
        with Image.open(frame_path) as image:
            assert np.asarray(image).std() > 5
            sheet.paste(image.resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
    sheet.save(s.OUT / "video_contact_sheet.png")
    s.write(s.OUT / "video_verification.json", {
        "passed": True, "seed": protocol["video_seed"], "controllers": names,
        "poses_from_evaluation_not_reselected": True, "all_recorded_control_tilts_match_raw_data": True,
        "decoded_frames": frames, "seconds": seconds, "fps": metadata["fps"], "size": metadata["size"],
        "all_frames_decoded": True, "silent": True,
        "font_file": str(media.FONT_DIR / "lmroman10-regular.otf"),
        "font_sha256": s.digest(media.FONT_DIR / "lmroman10-regular.otf"),
        "renderer_sha256": s.digest(HELPER), "video_sha256": s.digest(dest)})


if __name__ == "__main__":
    main()
