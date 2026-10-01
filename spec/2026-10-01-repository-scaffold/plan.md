# F0 · Repository scaffold: Plan

## Context and Design Decisions

- **Shared Hydra groups live in the library** (`lib/irwm/conf`) and each project's primary config pulls them in with `hydra.searchpath: [pkg://irwm.conf]`. The usual alternative is a top-level `conf/` folder found by a relative path. That breaks as soon as a script runs from another working directory (as SLURM jobs do), and it would mean copying the groups into every project. Putting them in the package respects "reusable logic goes in `lib/irwm`" (TechStack §2).
- **Heavy shared deps belong to `irwm`**, not to the projects. torch, lightning, hydra, wandb and submitit are needed by every project, so they go into `lib/pyproject.toml`. Project members declare only project-specific extras (none yet). Each later feature adds its own deps (`h5py`, `pycocotools`, …), which keeps F0's lock small.
- **Torch CPU/CUDA split** uses `[tool.uv.sources]` with platform markers and two explicit indexes (TechStack §1.1). `tool.uv.environments` limits resolution to win32 and linux-x86_64, because torch 2.2.0+cu121 has no aarch64 or macOS wheels and an unrestricted universal lock would fail. `numpy<2` is pinned because torch 2.2 wheels are compiled against NumPy 1.x.
- **`.env` loading is explicit and non-overriding.** `load_env()` is called at the top of every entry point, before `@hydra.main`, because Hydra resolves `hydra.run.dir` (which interpolates `${oc.env:IRWM_OUTPUT_ROOT}`) before the task function runs. `override=False` means anything set by the shell or sbatch wins, so the same `.env` can never redirect a cluster job.
- **`paths/default.yaml` has no defaults** for `IRWM_*`. A missing variable must fail loudly; a silent fallback would scatter caches across the filesystem (TechStack §6). W&B variables do get defaults (project name, online mode) because those are safe.
- **Resume semantics.** Lightning's `ckpt_path="last"` only finds checkpoints in the `ModelCheckpoint` dir. The smoke script passes the explicit `last.ckpt` path when it exists, so the behaviour stays the same whatever Lightning version is installed. Checkpoints go in a run-name-stable dir (`${paths.ckpt_root}/smoke/${run_name}`). On SLURM, `run_name` defaults to `slurm-${SLURM_JOB_ID}`, which stays the same across requeues, so a requeued job resumes. Locally it defaults to the timestamp.
- **Submitit vs sbatch.** Single jobs use `slurm/job.sbatch`; sweeps use `-m cluster=slurm` (submitit). Both read the same `cluster/slurm.yaml`. Selecting the submitit launcher when not in multirun mode has no effect, so a job run through sbatch with `cluster=slurm` behaves correctly too.
- **Smoke project is permanent** (user decision). It is the first thing to run after any env or cluster change.
- **Not installing DINO-Foresight.** F0 only pins the submodule. Whether the adapter imports it or vendors parts of it is decided in F1 and F10.

## Step 1 — Repo hygiene files
Create `.gitignore` (FR12), `.env.example`, `.python-version` (`3.11`).

## Step 2 — `lib/` workspace member
- `lib/pyproject.toml`: `name = "irwm"`, hatchling build, `packages = ["irwm"]`, deps: `torch==2.2.0`, `torchvision==0.17.0`, `numpy>=1.26,<2`, `lightning>=2.4,<2.6`, `hydra-core==1.3.2`, `hydra-submitit-launcher==1.2.0`, `wandb`, `python-dotenv`. Hatch includes the `conf/**/*.yaml` files inside the package automatically (editable install, so they are read straight from source).
- `lib/irwm/__init__.py` (`__version__ = "0.1.0"`), plus empty `__init__.py` in `data`, `features`, `detection`, `worldmodel`, `eval`, `utils`.

## Step 3 — `irwm.utils.paths`
```python
ROOT_ENV_VARS = {"data": "IRWM_DATA_ROOT", "cache": "IRWM_CACHE_ROOT",
                 "ckpt": "IRWM_CKPT_ROOT", "output": "IRWM_OUTPUT_ROOT"}
def repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "uv.lock").is_file(): return p
    raise FileNotFoundError(...)
def load_env(dotenv_path=None):
    path = Path(dotenv_path) if dotenv_path else repo_root() / ".env"
    if not path.is_file(): return False
    dotenv.load_dotenv(path, override=False); return True
def get_root(name):
    if name not in ROOT_ENV_VARS: raise ValueError
    var = ROOT_ENV_VARS[name]; val = os.environ.get(var, "").strip()
    if not val: raise MissingEnvVarError(f"{var} is not set ... see .env.example")
    return Path(val).expanduser().resolve()
```

## Step 4 — `irwm.utils.tracking` and `irwm.utils.slurm`
- `git_info`: `subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo)` and `git status --porcelain --untracked-files=no`, both wrapped in `try/except (OSError, CalledProcessError)`. The default `repo` is `repo_root()`, or the cwd if that fails.
- `make_wandb_logger(cfg, save_dir)`: `WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity, group=cfg.wandb.group, name=cfg.run_name, tags=list(cfg.wandb.tags), mode=cfg.wandb.mode, save_dir=str(save_dir), log_model=cfg.wandb.log_model)`, then `logger.experiment.config.update({...resolved cfg..., "git": git_info()}, allow_val_change=True)`. Done via `log_hyperparams` to stay rank-zero safe.
- `on_slurm`, `cluster_plugins` as specified in FR7.

## Step 5 — Hydra base configs
Files under `lib/irwm/conf/`: `irwm_base.yaml`, `paths/default.yaml`, `wandb/default.yaml`, `cluster/local.yaml`, `cluster/slurm.yaml`.
`cluster/slurm.yaml` is `# @package _global_` with:
```yaml
defaults:
  - override /hydra/launcher: submitit_slurm
cluster: {name: slurm, accelerator: gpu, devices: 1, num_nodes: 1, num_workers: 8,
          precision: 32-true, partition: TODO, account: null, qos: null, gres: "gpu:1",
          cpus_per_task: 8, mem_gb: 32, timeout_min: 240}
hydra:
  launcher:
    partition: ${cluster.partition}
    account: ${cluster.account}   # (passed via additional_parameters when null-safe)
    gres: ${cluster.gres}
    cpus_per_task: ${cluster.cpus_per_task}
    mem_gb: ${cluster.mem_gb}
    timeout_min: ${cluster.timeout_min}
    tasks_per_node: ${cluster.devices}
    nodes: ${cluster.num_nodes}
    submitit_folder: ${hydra.sweep.dir}/.submitit/%j
```
The `cluster` group is selected through `irwm_base`'s defaults, so `cluster=slurm` on the command line swaps it.

## Step 6 — Smoke project
- `projects/smoke/pyproject.toml`: `name = "irwm-smoke"`, `dependencies = ["irwm"]`, `[tool.uv] package = false`.
- `projects/smoke/conf/train.yaml`:
```yaml
defaults: [irwm_base, _self_]
hydra: {searchpath: ["pkg://irwm.conf"]}
project: smoke
run_name: ${oc.env:SLURM_JOB_ID,local}-${now:%Y%m%d-%H%M%S}   # stable-across-requeue variant set in sbatch
data: {n_samples: 256, in_dim: 8, batch_size: 32}
trainer: {max_epochs: 3}
lr: 0.05
```
  On SLURM, `job.sbatch` passes `run_name=slurm-$SLURM_JOB_ID` so requeues land in the same ckpt dir.
- `projects/smoke/train.py`: a `SmokeModel(LightningModule)` (a single `nn.Linear`, MSE loss), with a `make_loaders(cfg)` that uses a `torch.Generator` seeded by `cfg.seed` and `w_true` for the targets, and the main function:
```python
load_env()
@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg):
    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name
    logger = make_wandb_logger(cfg, save_dir=run_dir)
    trainer = L.Trainer(accelerator=cfg.cluster.accelerator, devices=..., num_nodes=..., precision=...,
                        max_epochs=..., logger=logger, plugins=cluster_plugins(),
                        callbacks=[ModelCheckpoint(dirpath=ckpt_dir, save_last=True)],
                        default_root_dir=run_dir, enable_progress_bar=not on_slurm())
    last = ckpt_dir / "last.ckpt"
    trainer.fit(model, train_dl, val_dl, ckpt_path=str(last) if last.exists() else None)
    if trainer.is_global_zero: write result.json
    return final_val_loss
```

## Step 7 — SLURM template
`slurm/job.sbatch` (FR9) and `slurm/README.md` (a few lines of usage).

## Step 8 — Root workspace + lock + sync
Root `pyproject.toml`: `[project] name = "ir-world-model"`, `requires-python`, `dependencies = ["irwm", "irwm-smoke"]`, `[dependency-groups] dev = ["pytest>=8"]`, `[tool.uv] environments`, `[tool.uv.workspace]`, `[tool.uv.sources]` (`irwm`/`irwm-smoke` = workspace, torch/torchvision split by marker), `[[tool.uv.index]]` × 2, `[tool.pytest.ini_options]` (FR10). Run `uv lock` and `uv sync`.

## Step 9 — Tests
`tests/conftest.py` sets `WANDB_MODE=disabled` and points the `IRWM_*` variables at `tmp_path` (fixture `irwm_env`). `tests/test_repo_scaffold.py` covers validation Groups 1–4 (below).

## Step 10 — Submodule
`git submodule add https://github.com/Sta8is/DINO-Foresight.git third_party/DINO-Foresight`, then check out `d98f765…` and commit the pinned pointer.

## Step 11 — README + constitution updates
- Short `README.md` covering setup (uv install, `.env`, `uv sync`, smoke run, server notes).
- Roadmap: F0 → ✓ DONE, or ◐ with the remaining server-side items listed.
- TechStack: Component Registry row for F0 → ✓; add rows for `irwm.utils.tracking`/`slurm`, the shared Hydra conf (`lib/irwm/conf`), `projects/smoke`, `slurm/job.sbatch`; add `projects/smoke/` to the §2 layout; note in §1/§2 that shared Hydra groups live in `pkg://irwm.conf`.

## Implementation Order
1. Hygiene files
2. `lib/` member skeleton
3. `paths.py`
4. `tracking.py`, `slurm.py`
5. Hydra base configs
6. Smoke project
7. sbatch template
8. Root pyproject → `uv lock` → `uv sync`
9. Tests → `uv run pytest`
10. Submodule
11. README + constitution updates
