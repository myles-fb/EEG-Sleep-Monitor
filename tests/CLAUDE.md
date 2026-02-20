# tests — Test Suite

> **Start here:** Read `docs/sessions/2026-02-20-dashboard-redesign.md` for the latest session context.

## Run Tests
```bash
.venv/bin/python -m pytest tests/ -v          # all tests
.venv/bin/python -m pytest tests/test_mos.py -v       # MOs pipeline (10 tests)
.venv/bin/python -m pytest tests/test_study_service.py -v  # windowing fix (3 tests)
```

**IMPORTANT:** Use `.venv/bin/python -m pytest`, NOT `.venv/bin/pytest` (import path issue).

## Test Files
- `test_mos.py` — 10 tests covering MOs pipeline end-to-end (all passing)
- `test_study_service.py` — 3 tests for `_compute_n_windows` windowing fix (added dashboard-redesign)
- `test_ring_buffer.py` — placeholder (no tests)
- `test_metrics.py` — placeholder (no tests)

## sys.path Setup (required in every test file)
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

## Current Status
13 total tests, 13 passing (10 original + 3 from dashboard-redesign windowing fix).

## Test Conventions
- Test function names: `test_<what_it_tests>()`
- Use minimal fixtures; avoid mocking the DB (use in-memory SQLite via `init_db()` if needed)
- For viz_helpers: test is a smoke import test + basic call (no assertion needed beyond no-crash)
