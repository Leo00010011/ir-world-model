"""Download and pin every DINO-Foresight / DINOv2 artefact into $IRWM_CKPT_ROOT.

Run on a LOGIN node (needs internet); idempotent:

    uv run python projects/dinof_audit/download.py

Verified files are skipped. The final manifests are what ``audit.py`` (offline, GPU) checks.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info
from irwm.worldmodel import assets as A

load_env()


def _manifest_ok(directory: Path, name: str) -> bool:
    row = A.read_manifest(directory).get(name)
    p = directory / name
    return bool(row) and p.is_file() and p.stat().st_size == row["bytes"] and A.sha256_file(p) == row["sha256"]


def download_hf(files: list[str]) -> list[dict]:
    from huggingface_hub import hf_hub_download

    out_dir = A.dinof_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in files:
        if _manifest_ok(out_dir, name):
            print(f"[hf] ok (cached)  {name}")
        else:
            print(f"[hf] downloading  {name}")
            hf_hub_download(A.DINOF_HF_REPO, name, revision=A.DINOF_HF_REVISION, local_dir=out_dir)
        entries.append({"file": name, "source": f"hf://{A.DINOF_HF_REPO}", "revision": A.DINOF_HF_REVISION})
    return entries


def download_dinov2_code(name: str) -> Path:
    sha = A.DINOV2_HUB_COMMITS[name]
    target = A.dinov2_code_dir(name)
    if (target / "hubconf.py").is_file():
        print(f"[dinov2] ok (cached) {name} @ {sha[:8]}")
        return target
    url = f"https://codeload.github.com/{A.DINOV2_REPO}/zip/{sha}"
    print(f"[dinov2] downloading {name} @ {sha[:8]}")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "code.zip"
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        (top,) = [p for p in Path(tmp).iterdir() if p.is_dir()]  # dinov2-<sha>/
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(top, target)
    (target / "COMMIT").write_text(sha + "\n")
    return target


def download_dinov2_weights() -> Path:
    import torch

    path = A.dinov2_weights_path()
    if _manifest_ok(A.dinov2_hub_dir(), path.relative_to(A.dinov2_hub_dir()).as_posix()):
        print(f"[dinov2] ok (cached) {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dinov2] downloading {path.name}")
    tmp = path.with_suffix(".part")
    torch.hub.download_url_to_file(A.DINOV2_WEIGHTS_URL, str(tmp), progress=True)
    tmp.replace(path)
    return path


def hub_manifest_entries() -> list[dict]:
    hub = A.dinov2_hub_dir()
    entries = [{"file": A.dinov2_weights_path().relative_to(hub).as_posix(), "source": A.DINOV2_WEIGHTS_URL}]
    code_dirs = [A.dinov2_code_dir(n) for n in A.DINOV2_HUB_COMMITS] + [hub / A.DINOV2_HUB_CACHE_NAME]
    for d in code_dirs:
        commit = (d / "COMMIT").read_text().strip()
        for f in sorted(p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            entries.append({"file": f.relative_to(hub).as_posix(), "source": f"github://{A.DINOV2_REPO}",
                            "revision": commit})
    return entries


@hydra.main(version_base="1.3", config_path="conf", config_name="download")
def main(cfg: DictConfig) -> None:
    from pca_export import export_pca

    report: dict = {"git": git_info(), "hf_revision": A.DINOF_HF_REVISION}

    # 1. Hugging Face files
    hf_entries = download_hf(list(cfg.hf_files))
    pca_src = A.dinof_dir() / A.DINOF_PCA
    report["pca_export"] = export_pca(pca_src, A.dinof_dir() / A.PCA_EXPORT)
    hf_entries.append({"file": A.PCA_EXPORT, "source": f"export of {A.DINOF_PCA}"})
    A.write_manifest(A.dinof_dir(), hf_entries)

    # 2. DINOv2 hub code (all audited versions) + weights + torch.hub cache alias of the pin
    for name in A.DINOV2_HUB_COMMITS:
        download_dinov2_code(name)
    alias = A.dinov2_hub_dir() / A.DINOV2_HUB_CACHE_NAME
    if alias.exists():
        shutil.rmtree(alias)
    shutil.copytree(A.dinov2_code_dir(A.DINOV2_HUB_PINNED), alias)
    download_dinov2_weights()
    A.write_manifest(A.dinov2_hub_dir(), hub_manifest_entries())

    report["verify"] = {"dinof_dir": A.verify_manifest(A.dinof_dir()),
                        "dinov2_hub_dir": A.verify_manifest(A.dinov2_hub_dir())}
    report["paths"] = {"dinof_dir": str(A.dinof_dir()), "dinov2_hub_dir": str(A.dinov2_hub_dir())}
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "download_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if any(report["verify"].values()):
        raise SystemExit(f"Manifest verification failed: {report['verify']}")


if __name__ == "__main__":
    main()
