"""Smoke test of the whole stack: Hydra → Lightning → W&B, resumable under SLURM.

Fits a linear regression on seeded random data. Run it after any environment or
cluster change:

    uv run python projects/smoke/train.py                      # local
    sbatch slurm/job.sbatch projects/smoke/train.py cluster=slurm
    uv run python projects/smoke/train.py -m cluster=slurm seed=0,1   # submitit sweep
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from irwm.utils.paths import load_env
from irwm.utils.slurm import cluster_plugins, on_slurm, resume_ckpt_path
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()


class SmokeModel(L.LightningModule):
    def __init__(self, in_dim: int, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, in_dim]`` → ``[B, 1]``."""
        return self.net(x)

    def _loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self._loss(batch), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def make_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Seeded synthetic data ``y = x @ w_true + noise``. Returns (train, val) loaders."""
    d = cfg.data
    g = torch.Generator().manual_seed(cfg.seed)
    x = torch.randn(d.n_samples, d.in_dim, generator=g)
    w_true = torch.randn(d.in_dim, 1, generator=g)
    y = x @ w_true + d.noise * torch.randn(d.n_samples, 1, generator=g)
    n_val = int(d.n_samples * d.val_fraction)
    kw = dict(batch_size=d.batch_size, num_workers=cfg.cluster.num_workers)
    train = DataLoader(TensorDataset(x[n_val:], y[n_val:]), shuffle=True, generator=g, **kw)
    val = DataLoader(TensorDataset(x[:n_val], y[:n_val]), **kw)
    return train, val


@torch.no_grad()
def eval_loss(model: SmokeModel, loader: DataLoader) -> float:
    was_training = model.training
    model.eval()
    losses = [nn.functional.mse_loss(model(x), y, reduction="sum") for x, y in loader]
    model.train(was_training)
    return float(torch.stack(losses).sum() / len(loader.dataset))


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> float:
    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # Stable across SLURM requeues (run_name = slurm-<job id>), unlike the timestamped run dir.
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name

    train_dl, val_dl = make_loaders(cfg)
    model = SmokeModel(cfg.data.in_dim, cfg.lr)
    initial_val_loss = eval_loss(model, val_dl)

    trainer = L.Trainer(
        accelerator=cfg.cluster.accelerator,
        devices=cfg.cluster.devices,
        num_nodes=cfg.cluster.num_nodes,
        precision=cfg.cluster.precision,
        max_epochs=cfg.trainer.max_epochs,
        logger=make_wandb_logger(cfg, save_dir=run_dir),
        plugins=cluster_plugins(),
        callbacks=[ModelCheckpoint(dirpath=ckpt_dir, save_last=True)],
        # SLURM auto-requeue writes hpc_ckpt_*.ckpt here and restores from it on restart.
        default_root_dir=ckpt_dir,
        enable_progress_bar=not on_slurm(),
        log_every_n_steps=1,
    )
    resume_from = resume_ckpt_path(ckpt_dir)
    trainer.fit(model, train_dl, val_dl, ckpt_path=resume_from)

    final_val_loss = eval_loss(model.cpu(), val_dl)
    if trainer.is_global_zero:
        result = {
            "initial_val_loss": initial_val_loss,
            "final_val_loss": final_val_loss,
            "epochs_completed": trainer.current_epoch,
            "resumed_from": resume_from,
            "ckpt_dir": str(ckpt_dir),
            "git": git_info(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    if not math.isfinite(final_val_loss):
        raise RuntimeError(f"Non-finite val loss: {final_val_loss}")
    return final_val_loss


if __name__ == "__main__":
    main()
