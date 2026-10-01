# F0 · Repository scaffold: Validation

## Code Correctness (`tests/test_repo_scaffold.py`)

### Group 1 — Package and environment
- [x] `import irwm` succeeds and `irwm.__version__` is a `str`; each of `irwm.{data,features,detection,worldmodel,eval,utils}` imports.
- [x] `sys.version_info[:2] == (3, 11)` inside the venv.
- [x] `torch.__version__` starts with `"2.2.0"`; `numpy.__version__` major is `1`.
- [x] `importlib.resources.files("irwm.conf")` contains `irwm_base.yaml`, `paths/default.yaml`, `wandb/default.yaml`, `cluster/local.yaml`, `cluster/slurm.yaml`.

### Group 2 — Path resolution
- [x] With all four `IRWM_*` set to tmp dirs, `get_root(n)` for each `n` in `ROOT_ENV_VARS` returns that dir as an absolute `Path`; the wrappers `data_root()` etc. return the same values.
- [x] `get_root("bogus")` raises `ValueError`.
- [x] With `IRWM_CACHE_ROOT` unset → `cache_root()` raises `MissingEnvVarError`, which `isinstance(..., KeyError)`, and the message contains `"IRWM_CACHE_ROOT"`. Same for a value of `"   "`.
- [x] `get_root` with `~/x` returns a path without `~` (expanded).
- [x] `repo_root()` contains `uv.lock` and `lib/irwm`.
- [x] `load_env(p)` for a tmp `.env` containing `IRWM_DATA_ROOT=/from/dotenv` and `IRWM_CKPT_ROOT=/from/dotenv`, with `IRWM_DATA_ROOT` already set to `/from/shell`: returns `True`; `IRWM_DATA_ROOT` stays `/from/shell` (no override); `IRWM_CKPT_ROOT == "/from/dotenv"`.
- [x] `load_env(tmp/"missing.env")` returns `False` and does not raise.
- [x] No file under `lib/irwm/` matches a hard-coded absolute path regex (`[A-Za-z]:\\\\|/home/|/scratch/|/cluster/`).

### Group 3 — Hydra composition
- [x] Composing `projects/smoke/conf/train.yaml` (via `initialize_config_dir` + `compose`) with `IRWM_*` set gives `cfg.paths.output_root == $IRWM_OUTPUT_ROOT`, `cfg.cluster.name == "local"`, `cfg.wandb.group == "smoke"`, `cfg.wandb.project == "ir-world-model"` (with `WANDB_PROJECT` unset).
- [x] With `IRWM_DATA_ROOT` unset, accessing `cfg.paths.data_root` raises (`InterpolationResolutionError`).
- [x] Composing with override `cluster=slurm` and `return_hydra_config=True`: `cfg.cluster.name == "slurm"`, `cfg.cluster.accelerator == "gpu"`, `cfg.hydra.launcher._target_` ends with `SlurmLauncher`, `cfg.hydra.launcher.gres == cfg.cluster.gres`.
- [x] `cfg.hydra.run.dir` (with `return_hydra_config=True`) starts with `${IRWM_OUTPUT_ROOT}/smoke/`.

### Group 4 — Tracking, SLURM helpers, smoke run
- [x] `git_info()` in the repo returns a 40-hex-char `commit` and a `bool` `dirty`; `git_info(tmp_path)` (not a repo) returns `{"commit": "unknown", "dirty": False}`.
- [x] `on_slurm()` is `False` without `SLURM_JOB_ID`, `True` with it; `cluster_plugins()` is `[]` / one `SLURMEnvironment` with `auto_requeue=True` respectively.
- [x] `resume_ckpt_path(d)`: empty dir → `None`; only `last.ckpt` → it; `last.ckpt` + `hpc_ckpt_1.ckpt` + `hpc_ckpt_10.ckpt` + `hpc_ckpt_x.ckpt` → `hpc_ckpt_10.ckpt` (numeric, not lexicographic, max; non-numeric ignored).
- [x] End-to-end: running `projects/smoke/train.py` as a subprocess with `WANDB_MODE=disabled`, `IRWM_*` in tmp, `trainer.max_epochs=2` exits 0 in < 60 s; `result.json` exists under `$IRWM_OUTPUT_ROOT/smoke/` with a finite `final_val_loss` and a `git.commit`; `last.ckpt` exists under `$IRWM_CKPT_ROOT/smoke/<run_name>/`.
- [x] Resume: re-running with the same `run_name` and `trainer.max_epochs=4` exits 0 and the second `result.json` reports `epochs_completed == 4` and `resumed_from` ending in `last.ckpt`; the first run's `resumed_from` is `null`.
- [x] Learning sanity: `final_val_loss` after 4 epochs < the val loss at initialisation (also recorded in `result.json` as `initial_val_loss`).

### Group 5 — pytest configuration
- [x] A test marked `@pytest.mark.data` exists and is reported as deselected by a plain `uv run pytest`; whole suite runs < 60 s on CPU.

## Data Validity
No dataset is touched in F0. In its place, these checks confirm the environment:
- [x] `uv lock` resolves for both environments. `uv.lock` contains a `torch` entry with a `+cu121` wheel for `linux_x86_64` and a `+cpu` wheel for `win_amd64`.
- [x] `uv sync` on a clean `.venv` succeeds locally (Windows).
- [ ] **Manual (user, server):** `pip install --user uv && uv sync --frozen` on the login node, then `uv run python -c "import torch; print(torch.cuda.is_available())"` prints `True` on a GPU node.
- [ ] **Manual (user, W&B online):** a local `uv run python projects/smoke/train.py` with `WANDB_MODE=online` shows a run in project `ir-world-model`, group `smoke`, whose config contains `git.commit`.
- [ ] **Manual (user, server):** after the TODOs in `cluster/slurm.yaml` and `slurm/job.sbatch` are filled in, `sbatch slurm/job.sbatch projects/smoke/train.py cluster=slurm` completes, logs to W&B, and writes `result.json`. Optional: `scontrol requeue <jobid>` mid-run, then confirm the job resumes from `last.ckpt`.
- [ ] **Manual (user, server):** `uv run python projects/smoke/train.py -m cluster=slurm seed=0,1` submits two jobs through submitit.

## Data Architecture Integrity
- [x] Every path the smoke run writes is under `$IRWM_OUTPUT_ROOT` or `$IRWM_CKPT_ROOT`. The test checks that nothing new appears in the repo's working tree apart from git-ignored paths: `git status --porcelain` is unchanged before and after the smoke subprocess.
- [x] `lib/irwm` imports nothing from `projects/` (grep check in the tests).
- [x] `third_party/DINO-Foresight` is at commit `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0` (`git submodule status`), and its files are not modified (`git -C third_party/DINO-Foresight status --porcelain` is empty).
- [x] pytest does not collect anything under `third_party/` (`testpaths = ["tests"]`).
