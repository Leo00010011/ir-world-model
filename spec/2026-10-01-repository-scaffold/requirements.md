# F0 · Repository scaffold: Requirements

## Goal

Set up the monorepo that every later feature builds on. That means a uv workspace that gives the same environment on the Windows laptop and the Linux SLURM cluster, a shared `irwm` package that resolves every filesystem path from environment variables, base Hydra config groups (`paths`, `cluster`, `wandb`) that every project reuses, a SLURM sbatch template plus a submitit launcher config, a pytest setup with a `data` marker that is skipped by default, and DINO-Foresight pinned as a git submodule. A small smoke project (`projects/smoke/`) checks the whole stack end to end: Hydra, Lightning, W&B, and SLURM requeue. It stays in the repo as a permanent health check for the environment and the cluster.

## Scope

In scope:
- Root `pyproject.toml` (uv workspace, members `lib` and `projects/*`), `.python-version` (3.11), `uv.lock`.
- `lib/` workspace member providing the editable package `irwm` with empty sub-packages `data`, `features`, `detection`, `worldmodel`, `eval`, `utils`.
- `irwm.utils.paths`: env-var path resolution and `.env` loading.
- `irwm.utils.tracking`: git commit info and W&B logger construction.
- `irwm.utils.slurm`: SLURM detection and Lightning cluster plugins (auto-requeue).
- Shared Hydra configs in `lib/irwm/conf/` (`irwm_base`, `paths/default`, `cluster/{local,slurm}`, `wandb/default`), discovered with `hydra.searchpath: [pkg://irwm.conf]`.
- `projects/smoke/`: a `@hydra.main` entry point that trains a tiny Lightning model on random tensors and logs to W&B.
- `slurm/job.sbatch` (generic single-job template) and the submitit launcher settings in `cluster/slurm.yaml`.
- pytest config (`testpaths`, `data` marker, default `-m "not data"`) and `tests/test_repo_scaffold.py`.
- `third_party/DINO-Foresight` submodule pinned at `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`.
- `.gitignore`, `.env.example`, a short `README.md` (setup instructions).

Out of scope:
- Installing or importing DINO-Foresight's code or weights (F1).
- `h5py`, `pycocotools`, `optuna`, and the other deps of later features. Each feature adds what it needs.
- Folders for the real projects (`feature_extraction`, `detection`, `wm_eval`, `wm_adaptation`). Each one is created by the first feature that needs it.
- Real cluster values (partition, account, gres). These are placeholders marked `TODO` for the user to fill in.

## Functional Requirements

**FR1 — Workspace.** The root `pyproject.toml` declares `[tool.uv.workspace] members = ["lib", "projects/*"]`, `requires-python = ">=3.11,<3.12"`, and a root project that depends on `irwm`, `irwm-smoke` and a `dev` dependency group (`pytest`). `uv sync` on a clean checkout creates `.venv` with Python 3.11. `uv lock` resolves for exactly two environments: `sys_platform == 'win32'` and `sys_platform == 'linux' and platform_machine == 'x86_64'`.

**FR2 — Torch source split.** `torch==2.2.0` and `torchvision==0.17.0` resolve from `https://download.pytorch.org/whl/cu121` on Linux and from `https://download.pytorch.org/whl/cpu` on Windows. Both indexes are `explicit = true`, so no other package is taken from them. `numpy<2` is pinned because torch 2.2 is built against NumPy 1.x.

**FR3 — `irwm` package.** `import irwm` works from the venv, and `irwm.__version__` is a string. The sub-packages `irwm.{data,features,detection,worldmodel,eval,utils}` are importable. `lib/irwm/conf/**/*.yaml` is shipped with the package, so `pkg://irwm.conf` resolves.

**FR4 — Path resolution (`irwm.utils.paths`).**
- `ROOT_ENV_VARS: dict[str, str]` maps the logical names `data`, `cache`, `ckpt`, `output` to `IRWM_DATA_ROOT`, `IRWM_CACHE_ROOT`, `IRWM_CKPT_ROOT`, `IRWM_OUTPUT_ROOT`.
- `get_root(name)` returns an absolute, `~`-expanded `pathlib.Path` and does not require the path to exist.
  - Unknown `name` → `ValueError`.
  - Variable unset or whitespace-only → `MissingEnvVarError` (a `KeyError` subclass). The message names the variable and points to `.env.example`.
- `data_root()`, `cache_root()`, `ckpt_root()`, `output_root()` are thin wrappers.
- `repo_root()` returns the nearest ancestor of `paths.py` that contains `uv.lock`. If none is found it raises `FileNotFoundError`.
- `load_env(dotenv_path=None) -> bool` loads `repo_root()/.env` (or the given path) with `override=False`, so variables already set in the process (for example by SLURM) always win. It returns `True` if a file was loaded. A missing file is not an error.
- No module in `lib/irwm` contains a hard-coded absolute path.

**FR5 — Hydra base configs** (in `lib/irwm/conf/`):
- `paths/default.yaml`: `data_root`, `cache_root`, `ckpt_root`, `output_root`, each set to `${oc.env:IRWM_*}` with no default, so composing without the variable fails loudly when the value is accessed.
- `wandb/default.yaml`: `project: ${oc.env:WANDB_PROJECT,ir-world-model}`, `entity: ${oc.env:WANDB_ENTITY,null}`, `mode: ${oc.env:WANDB_MODE,online}`, `group: ${project}`, `tags: []`, `log_model: false`.
- `cluster/local.yaml`: `name: local`, `accelerator: auto`, `devices: 1`, `num_nodes: 1`, `num_workers: 0`, `precision: 32-true`.
- `cluster/slurm.yaml`: `name: slurm`, `accelerator: gpu`, `devices`, `num_nodes`, `num_workers`, `precision`, and SLURM resources (`partition`, `account`, `qos`, `gres`, `cpus_per_task`, `mem_gb`, `timeout_min`), with placeholders marked `# TODO`. It also selects `hydra/launcher: submitit_slurm` and maps those resources onto `hydra.launcher.*`. The launcher's `submitit_folder` lives under the Hydra sweep dir.
- `irwm_base.yaml`: defaults `paths: default`, `cluster: local`, `wandb: default`. Requires the primary config to define `project`, `seed` (default 0 here). Sets `hydra.run.dir = ${paths.output_root}/${project}/${now:%Y-%m-%d}/${now:%H-%M-%S}` and `hydra.sweep.dir = ${paths.output_root}/${project}/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}`.

**FR6 — Tracking (`irwm.utils.tracking`).**
- `git_info(repo=None) -> dict` returns `{"commit": str, "dirty": bool}`. Outside a git repo, or without git, it returns `{"commit": "unknown", "dirty": False}` and does not raise.
- `make_wandb_logger(cfg, save_dir) -> WandbLogger` builds a Lightning `WandbLogger` from `cfg.wandb`, using `cfg.project` as the group, the run name derived from the Hydra run dir, and `config = resolved cfg + {"git": git_info()}`.

**FR7 — SLURM helpers (`irwm.utils.slurm`).**
- `on_slurm() -> bool` is true iff `SLURM_JOB_ID` is set.
- `cluster_plugins() -> list` returns `[SLURMEnvironment(auto_requeue=True)]` on SLURM and `[]` otherwise.
- `resume_ckpt_path(ckpt_dir) -> str | None` returns the newest `hpc_ckpt_<n>.ckpt` (by integer `n`) if there is one, else `last.ckpt` if it exists, else `None`. *(Added during build: Lightning only auto-restores hpc checkpoints when `ckpt_path is None`, so passing `last.ckpt` explicitly would silently drop the newer mid-epoch checkpoint written at preemption.)*

**FR8 — Smoke project (`projects/smoke/`).** It is a workspace member (`irwm-smoke`, non-package) with `train.py` and `conf/train.yaml`. `train.py`:
- calls `load_env()` at import time, before `@hydra.main`
- seeds with `lightning.seed_everything(cfg.seed)`
- fits a linear regression `LightningModule` on a seeded random `TensorDataset` (`n_samples × in_dim` float32) for `trainer.max_epochs`
- logs `train/loss` and `val/loss`
- saves `last.ckpt` under `${paths.ckpt_root}/smoke/<run_name>/` and resumes via `resume_ckpt_path(ckpt_dir)`; `default_root_dir = ckpt_dir` so SLURM hpc checkpoints land there too
- writes `result.json` (`final_val_loss`, `git`, resolved `config`) in the Hydra run dir

With `cluster=local` it runs on CPU in under 30 s.

**FR9 — SLURM template (`slurm/job.sbatch`).** A generic template that takes the entry point and Hydra overrides as arguments (`sbatch slurm/job.sbatch projects/smoke/train.py cluster=slurm`). It sets `--requeue`, `--signal=SIGUSR1@90`, and `--output` relative to the submit dir. It contains no absolute paths; the repo is the submit dir. It sources `.env` if one is present, then runs `uv run --frozen python "$@"` (or `.venv/bin/python` when `IRWM_NO_UV_RUN=1`, for offline nodes). Resources come from `#SBATCH` lines marked TODO, mirrored in `cluster/slurm.yaml`.

**FR10 — pytest.** `[tool.pytest.ini_options]` has `testpaths = ["tests"]`, `markers = ["data: needs the real dataset/weights; skipped by default"]`, and `addopts = "-m 'not data'"`. `uv run pytest` passes on CPU in under 60 s without network access, with W&B in `disabled` mode during tests.

**FR11 — Submodule.** `third_party/DINO-Foresight` is a git submodule of `https://github.com/Sta8is/DINO-Foresight.git` at commit `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`. It is not a workspace member and not on `sys.path`.

**FR12 — Hygiene.**
- `.gitignore` excludes `.venv/`, `__pycache__/`, `.env`, `outputs/`, `multirun/`, `wandb/`, `lightning_logs/`, `.pytest_cache/`, `*.ckpt`, `*.h5`, `*.pth`, `slurm-*.out`, `slurm_logs/`.
- `.env.example` lists every variable in TechStack §6.

**FR13 — Multi-cluster support** *(added 2026-10-01, after the first build, at the user's request: several clusters, W&B key in a file)*.
- Resources are defined once per cluster in `lib/irwm/conf/cluster/<site>.yaml` (`# @package _global_`, `defaults: [slurm, _self_]`). `cluster/slurm.yaml` becomes a generic base with no TODOs. `example_site.yaml` is the template.
- `irwm.utils.slurm.sbatch_args(cluster) -> list[str]` maps `partition, account, qos, constraint, gres, num_nodes→--nodes, devices→--ntasks-per-node, cpus_per_task, mem_gb→--mem=<n>G, timeout_min→--time` and omits null/empty values, then appends `extra_sbatch_args` verbatim. `python -m irwm.utils.slurm <site> [overrides]` prints the flags, applying only `cluster.*` overrides. It exits 1 if the site still contains `TODO`, and 2 on bad usage.
- `slurm/env.sh` (sourced): puts `~/.local/bin` on PATH, loads `.env` (handles `export`, quotes, inline comments, CRLF, `~`), and exports `WANDB_API_KEY` from `${WANDB_API_KEY_FILE:-~/.wandb_api_key}` if unset. It never overrides variables already set. `job.sbatch`, `submit.sh`, `setup_env.sh` and submitit (`hydra.launcher.setup`) all source it.
- `slurm/submit.sh <site> <entry.py> [overrides]`: `sbatch --job-name=irwm-<project> --output=$IRWM_OUTPUT_ROOT/slurm_logs/%x-%j.out <flags> $SBATCH_EXTRA slurm/job.sbatch <entry.py> cluster=<site> [overrides]`. Compatible with bash < 4.4.
- `job.sbatch` keeps only the generic `#SBATCH` lines (`--requeue`, `--signal=SIGUSR1@90`, output, open-mode) and prints the W&B key presence and `nvidia-smi -L`.
- `slurm/setup_env.sh [--test]`: idempotent per-cluster bootstrap (submodules, uv install via `pip --user` with the standalone installer as fallback, `uv sync --frozen`, `.env` creation, `IRWM_*` dir creation, version/CUDA/driver report, W&B key verification, optional pytest). It exits non-zero if an `IRWM_*` variable is missing.
- `load_env()` also expands a leading `~` in `IRWM_*` values, because Hydra's `oc.env` passes them through raw.

## Public API Summary

```python
# irwm/utils/paths.py
ROOT_ENV_VARS: dict[str, str]
class MissingEnvVarError(KeyError): ...
def repo_root() -> Path: ...
def load_env(dotenv_path: str | Path | None = None) -> bool: ...
def get_root(name: str) -> Path: ...
def data_root() -> Path: ...
def cache_root() -> Path: ...
def ckpt_root() -> Path: ...
def output_root() -> Path: ...

# irwm/utils/tracking.py
def git_info(repo: str | Path | None = None) -> dict[str, str | bool]: ...
def make_wandb_logger(cfg: DictConfig, save_dir: str | Path) -> WandbLogger: ...

# irwm/utils/slurm.py
def on_slurm() -> bool: ...
def cluster_plugins() -> list: ...
def resume_ckpt_path(ckpt_dir: str | Path) -> str | None: ...
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `IRWM_*`, `WANDB_*` env vars | read | via `.env` locally, the shell/sbatch on the cluster |
| PyPI, download.pytorch.org (cpu, cu121) | read | at `uv lock`/`uv sync` time only |
| github.com/Sta8is/DINO-Foresight | read | submodule, pinned commit |
| `$IRWM_OUTPUT_ROOT/smoke/...` | write | Hydra run dirs, `result.json` |
| `$IRWM_CKPT_ROOT/smoke/...` | write | `last.ckpt` |
| W&B project `ir-world-model`, group `smoke` | write | smoke runs |
