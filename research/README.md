# Research source

Research modules live here; regression tests live in `../tests/`. Modules retain their original import names to keep historical code and notebooks compatible.

After installing dependencies, install this project editable from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .
```

Run a script explicitly, for example `python research/make_four_panel_demo.py --help`. Module imports such as `import diagnose_actions` resolve to this folder after the editable install. Outputs, model assets and traces continue to use the repository-root `results/`, `vendor/` and `work/` directories.

File movement and project-path repairs do not change controller formulas, mission criteria or reward weights. Historical source hashes record the old file layout. Their frozen audit manifests must not be silently rewritten to claim that these moved and edited sources are byte-identical to the original versions. Pre-move source backups remain in the ignored local work archive.
