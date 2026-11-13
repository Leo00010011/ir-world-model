"""Pinned MAE pre-trained ViT-B/16 weights for the end-to-end ViTDet ``mae`` detector.

detrex's ``dino_vitdet_base_4scale_12ep.py`` sets
``init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"``; this is the
same file, SHA-256-pinned so GPU jobs (compute nodes may be offline) never touch the network.
Download it once on a login node::

    uv run --frozen python -m irwm.detection.mae_assets

Layout::

    $IRWM_CKPT_ROOT/pretrained/mae/mae_pretrain_vit_base.pth
    $IRWM_CKPT_ROOT/pretrained/mae/manifest.json
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from irwm.utils.paths import ckpt_root
from irwm.worldmodel.assets import sha256_file, write_manifest

MAE_VITB_URL = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"
MAE_VITB_FILE = "mae_pretrain_vit_base.pth"
MAE_VITB_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
MAE_VITB_BYTES = 343249461


def mae_dir() -> Path:
    """``$IRWM_CKPT_ROOT/pretrained/mae``."""
    return ckpt_root() / "pretrained" / "mae"


def mae_weights_path() -> Path:
    """Where the pinned MAE ViT-B/16 checkpoint lives."""
    return mae_dir() / MAE_VITB_FILE


def verify_mae_weights(path: str | Path | None = None) -> Path:
    """Return ``path`` (default: the pinned location) after checking size and SHA-256; raise otherwise."""
    p = Path(path) if path is not None else mae_weights_path()
    if not p.is_file():
        raise FileNotFoundError(f"MAE weights not found at {p}; run `uv run --frozen python -m "
                                f"irwm.detection.mae_assets` on a node with internet")
    if p.stat().st_size != MAE_VITB_BYTES or sha256_file(p) != MAE_VITB_SHA256:
        raise ValueError(f"{p} does not match the pinned SHA-256 {MAE_VITB_SHA256}")
    return p


def download_mae_weights(force: bool = False) -> Path:
    """Download the MAE checkpoint into ``mae_dir()``, verify it and write ``manifest.json`` (login node)."""
    out = mae_weights_path()
    if out.is_file() and not force:
        verify_mae_weights(out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".part")
        urllib.request.urlretrieve(MAE_VITB_URL, part)  # noqa: S310 (fixed https URL)
        if sha256_file(part) != MAE_VITB_SHA256:
            part.unlink()
            raise ValueError(f"downloaded {MAE_VITB_URL} does not match the pinned SHA-256")
        part.replace(out)
    write_manifest(out.parent, [{"file": MAE_VITB_FILE, "source": MAE_VITB_URL}])
    return out


def main() -> None:
    from irwm.utils.paths import load_env

    load_env()  # a plain `python -m` run does not go through slurm/env.sh or a Hydra entry point
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the file verifies")
    print(download_mae_weights(force=parser.parse_args().force))


if __name__ == "__main__":
    main()
