Environment setup (Phase 6 reproducibility)

This project requires a Python environment with pinned dependencies.
Use the following minimal steps to create a reproducible environment locally or in CI.

1) Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix/macOS
source .venv/bin/activate
```

2) Install pinned dependencies

```bash
pip install --upgrade pip
pip install -r requirements-lock.txt
```

3) Run the smoke test (local)

```bash
python tests/smoke_test.py
```

Notes:
- `requirements-lock.txt` is a minimal locked set for CI smoke tests. For development you may want the unpinned `requirements.txt` (if present) or use `pip-tools` / `poetry` for reproducible builds.
- For CI, use Python 3.11 or 3.10 for best compatibility with these pinned packages.
- If plotting fails in headless CI, `tests/smoke_test.py` sets the Matplotlib backend to `Agg` to allow non-interactive plotting.
