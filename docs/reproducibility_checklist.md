Reproducibility Checklist

This checklist enumerates concrete actions and artifacts required to reproduce experimental results and analysis.

Environment Setup

- Use a fresh virtual environment; example (Windows PowerShell):

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Record interpreter path and package versions in `run_metadata.json` (e.g., output of `pip freeze`).

Dependency Control

- Use `requirements.txt` for pinned versions; consider `pip freeze > requirements.txt` at release time.
- For heavier reproducibility guarantees, provide a `pyproject.toml` or `conda` environment export.

Seed Control

- Seeds used for dataset splitting, model training, and analysis must be recorded and applied consistently. Analysis entrypoints accept `random_state` parameters for bootstrapping and Monte Carlo tests.
- Save `seed_list` and RNG configuration in `run_metadata.json`.

Artifact Storage

- Store all intermediate artifacts in `results/<dataset>/`.
- Keep copies of `raw_results.csv`, `normalized_results.csv`, `cbs_scores.csv`, and all analysis outputs (`cbs_validation/`, `significance/`).
- Commit analysis scripts and documentation in the repository; store large artifacts externally (artifact registry, Zenodo, OSF) and record their checksums in `run_metadata.json`.

Experiment Recreation Process

1. Checkout the reported `git_commit`.
2. Create and activate a virtual environment and install pinned dependencies.
3. Recreate or download dataset artifacts and verify checksums.
4. Run the data-preparation pipeline with the recorded preprocessing config.
5. Run model training and per-fold evaluation or, when available, re-run analysis directly on persisted `raw_results.csv` using the analysis CLI.
6. Compare generated artifacts with stored CSVs and plots.

Verification

- Include a smoke test script (example `scripts/verify_run.py`) that validates presence and basic sanity of key artifacts and returns non-zero exit code on mismatch.

Notes

- Where exact binary reproducibility is impossible (non-deterministic GPU ops), document acceptable tolerances for numeric differences and prefer seeded CPU runs when exact reproducibility is required for publication.