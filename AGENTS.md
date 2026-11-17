<!-- FOR AI AGENTS - Human readability is a side effect, not a goal -->
<!-- Managed by agent: keep sections and order; edit content, not structure -->
<!-- Last updated: 2026-11-05 | Last verified: 2026-11-05 -->

# AGENTS.md

**Precedence:** the closest `AGENTS.md` to the files you change wins; this root file is the repo-wide default.

## Environment: Nix flake + `nix develop` — READ BEFORE RUNNING PYTHON

Python and all dependencies come from the flake (`flake.nix`, built with uv2nix) — **not**
the system interpreter and **not** a `pip`/`uv run` virtualenv.

| Fact | Value |
|------|-------|
| Interpreter | nix-provided `python` (CPython 3.13.15) |
| pandas / numpy | `3.1.1` / `2.5.4` (from the nix store) |
| Shell entry | `nix develop`; `direnv` auto-loads via `.envrc` (fish and bash hooks are configured) |
| Runtime | the shell exposes a ready virtualenv — just call `python` |
| Non-NixOS hosts | `~/.local/bin/nix` uses native Nix when present, otherwise the rootless `nix-portable` fallback |
| Rootless direnv | `.envrc` exposes flake command shims because `nix-portable` virtualizes `/nix/store`; native Nix hosts use `use flake` directly |
| Python watcher | the dev-shell hook runs edited `*.py` files in eligible empty terminals; rootless `.envrc` also enters that hook |

Scripts read their `*.csv` by relative path, so run from the data's directory:

```bash
cd /home/johnmich/Work/Coding/COMP2211
nix develop
cd lab2 && python lab2_tasks.py
```

One-shot from the repo root: `nix develop -c sh -c 'cd lab2 && python lab2_tasks.py'`

**Do NOT use `uv run` / `uv sync` / `pip install` here.**
- The flake sets `UV_NO_SYNC=1`, `UV_PYTHON_DOWNLOADS=never`, and pins `UV_PYTHON` on
  purpose; `uv` is present only for editing `uv.lock`.
- `uv run`/`uv sync` build a `.venv` from PyPI wheels, which fail on NixOS with
  `ImportError: libstdc++.so.6: cannot open shared object file`.
- The system `python3` has no pandas/numpy; `pip` into it is unmanaged.

## Commands (verified 2026-11-05)
| Task | Command |
|------|---------|
| Enter shell | `nix develop` (or let `direnv` auto-load) |
| Versions | `python -c "import pandas,numpy;print(pandas.__version__,numpy.__version__)"` -> `3.1.1 2.5.4` |
| Run lab2 | `nix develop -c sh -c 'cd lab2 && python lab2_tasks.py'` |

> No `pytest`/`ruff`/`mypy`/`jupyter`/`ipython` are installed — don't invent those commands.

## Boundaries
- **Always:** run inside `nix develop`/direnv, never the system `python`; keep `lab2_tasks.py` and `lab2_tasks.ipynb` in sync.
- **Never:** `pip install` into the system interpreter, rely on a repo-local `.venv`, or commit `__pycache__/`/`.venv/`/`.direnv/`.
