# ir-world-model

Do latent world-model predictions keep the representational quality of real encoder features under an IR domain shift? We test this with DINO-Foresight (zero-shot) on FLIR ADAS v2. See [spec/constitution/Mission.md](spec/constitution/Mission.md) for the research plan and [spec/constitution/Roadmap.md](spec/constitution/Roadmap.md) for the feature plan.

## Setup

```bash
git clone --recurse-submodules <repo-url> && cd ir-world-model
# (existing clone: git submodule update --init)

pip install --user uv          # no admin rights needed; make sure its Scripts/bin dir is on PATH
uv sync                        # Python 3.11 + CPU torch (Windows) or cu121 torch (Linux)
cp .env.example .env           # then edit the IRWM_* paths
wandb login                    # once per machine (or WANDB_MODE=offline)
```

Check that everything works:

```bash
uv run pytest                              # CPU only, < 1 min; `-m data` runs the real-data tests
uv run python projects/smoke/train.py      # Hydra + Lightning + W&B end to end
```

On Windows, `py -m pytest` also works once the venv is active (`.venv\Scripts\activate`).

## Cluster (SLURM)

1. Once per cluster: copy [example_site.yaml](lib/irwm/conf/cluster/example_site.yaml) to `cluster/<site>.yaml`, fill in partition/account/gres, and commit.
2. On the login node, run `bash slurm/setup_env.sh --test`. It installs uv if needed, syncs the env, and checks torch/CUDA, the paths and the W&B key. Put the W&B key in `~/.wandb_api_key`.
3. Submit with `slurm/submit.sh <site> projects/smoke/train.py`, or run a sweep with `uv run python projects/smoke/train.py -m cluster=<site> seed=0,1`.

More details are in [slurm/README.md](slurm/README.md).

## Layout

| Path | Contents |
|---|---|
| `lib/irwm/` | Shared library (editable package `irwm`), including shared Hydra groups in `irwm/conf/` |
| `projects/<name>/` | Hydra entry points and configs only (`conf/` + `*.py`) |
| `third_party/DINO-Foresight/` | Pinned submodule, never modified |
| `slurm/` | sbatch templates |
| `spec/` | Constitution and one folder per feature |
| `tests/` | `test_<feature>.py` |

Every path comes from `IRWM_DATA_ROOT`, `IRWM_CACHE_ROOT`, `IRWM_CKPT_ROOT` and `IRWM_OUTPUT_ROOT`. Nothing is written into the repo.
