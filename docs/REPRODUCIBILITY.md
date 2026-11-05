# Reproducibility and repository scope

This repository publishes research source, a portable development-results snapshot and one demo video. The raw experiment archive is kept locally rather than committed. A fresh clone is sufficient to inspect the source, video and reported results; it does not contain everything needed to retrain the controller or rerender the historical episodes.

## Runtime and model

The recorded Windows environment uses Python 3.12, CPU PyTorch, MuJoCo 3.17.0 and Stable-Baselines3 2.7.1. `requirements.txt` lists the original direct simulation/training dependencies. `requirements-lock.txt` records the installed runtime, including analysis/rendering libraries. The local project was installed in editable mode and the selected regression checks passed. A fresh-clone setup was not tested.

The original setup uses uv:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt --torch-backend cpu
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie sparse-checkout set unitree_go2
git -C vendor/mujoco_menagerie fetch --depth 1 origin f3b6013629c7b94f8ee93fd977b161fa8350f9bc
git -C vendor/mujoco_menagerie checkout --detach f3b6013629c7b94f8ee93fd977b161fa8350f9bc
```

Do not clone over an existing vendor directory. Third-party model assets and policy weights are not included. Preserve upstream license notices when obtaining them.

The nominal Go2 RL Lab policy is from `wty-yy/go2_rl_lab`, commit `6c95d9feef4814c25eb6536e61bb38e1d1d718d4`, with checkpoint SHA256 `bde5210a8114aa08bb0c611e49def1482faa1b2441673e5681374c30b935e32e`. Its adapter expects `vendor/walking_baselines/go2_rl_lab/policy.pt` and the matching local source contract. Earlier Genesis experiments use a different adapter and checkpoint; they are not the retained walking candidate.

## Research code

Start with the environment, navigation, baseline and reviewer modules in `research/`. Earlier standing-control experiments are grouped in that same source folder; tests are in `tests/`. Their results and criteria must not be substituted for the later walking mission. `pyproject.toml` installs these flat modules without changing their established import names. Direct script execution now uses paths such as `python research/make_four_panel_demo.py --help`.

Historical runners depend on archived protocols, checkpoints and traces under `results/`, `vendor/` and a separate local experiment directory. Some scripts retain machine-specific paths to preserve their recorded versions and audit hashes. They are research artifacts, not a portable command-line package. Experiment tests can also require those ignored artifacts; a fresh clone should not be expected to pass the complete historical test suite without them.

Do not rerun an experiment's main function just to view its results. Runners write result directories and generally refuse overwrites. Inspect the portable `docs/results.json` snapshot instead.

## Video

The only retained export is `results/go2_four_panel_demo/go2_all_motors_four_panel_60s.mp4`. It shows all thirteen conditions at development pose 23003, with healthy control first and four simultaneous leg panels for hip, thigh and knee faults. It is not a montage of all 65 development episodes.

`research/make_four_panel_demo.py` records the final composition. Its renderer reuses earlier media helpers and requires the original saved qpos/qvel traces, local fonts, model assets and helper scripts. These dependencies are not bundled in this repository. No new rollout, training or policy inference was used to create the delivered demo.

The directory cleanup moved source and tests and repaired project paths. Recorded source hashes from earlier frozen studies still describe the pre-move versions. `research/project_paths.py` resolves relocated Python filenames in dynamic source lookups, but never bypasses checksum comparisons or treats a missing file as verified. Historical manifests were not rewritten to claim that changed source files match their old hashes. Repeating those original source-hash audits requires the original source snapshot, retained in the ignored local work archive.

At the owner's request, all twelve other exported videos were deleted from the local project during cleanup. Historical video plans and preservation manifests can still reference those deleted exports. Such manifests describe their original production state and cannot be rerun unchanged as post-cleanup preservation checks. The selected video's checksum and source-record coverage are preserved in `docs/results.json`; its MP4 content was not changed by cleanup.

## Evidence and limitations

The development snapshot is derived from the retained 65-record candidate index and checked against its audit summary and trace hashes. It includes one healthy condition and twelve motor conditions, five poses each. Some records reuse earlier development traces. It is not a fresh full-grid rerun, a held-out reliability estimate, or physical robot validation.

The controller uses simulator-provided fault information and condition-specific adjustments. Full-candidate validation still fails because four front hip/knee conditions are below target. New-seed final evaluation remains closed for this candidate.
