"""Run the *unmodified* DINO-Foresight reference code (third_party/DINO-Foresight).

This is the ground truth that our own encoder (F4) and adapter (F10) are tested against.
Heavy, reference-only dependencies (``pytorch_lightning``, ``timm``, ``einops``,
``sklearn``) are imported lazily inside functions. Install them with the ``irwm[reference]``
extra.

Offline and pinned: ``Dino_f.__init__`` calls ``torch.hub.load('facebookresearch/dinov2', ...)``,
which resolves the *unpinned* GitHub ``main`` at runtime. ``pinned_dinov2_hub`` redirects that
call, inside this process only, to the pinned local copy from ``irwm.worldmodel.assets``.
The submodule's files are never modified.
"""

from __future__ import annotations

import copy
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from irwm.utils.paths import repo_root
from irwm.worldmodel.assets import (
    DINOV2_HUB_PINNED,
    DINOV2_REPO,
    dinov2_code_dir,
    dinov2_hub_dir,
    dinov2_weights_path,
)

SUBMODULE = repo_root() / "third_party" / "DINO-Foresight"
DEMO_DIR = SUBMODULE / ".assets" / "demo" / "frankfurt_01_017101"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Overrides that turn the saved training args into a plain single-model inference setup.
_INFERENCE_OVERRIDES: dict[str, Any] = {
    "eval_mode": False,
    "eval_modality": None,
    "head_ckpt": None,
    "evaluate": False,
    "vis_attn": False,
}

CkptLike = str | Path | dict


def _load_ckpt(ckpt: CkptLike) -> dict:
    if isinstance(ckpt, dict):
        return ckpt
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def checkpoint_args(ckpt: CkptLike) -> Namespace:
    """The argparse namespace saved by ``Dino_f.save_hyperparameters()`` in the checkpoint."""
    return _load_ckpt(ckpt)["hyper_parameters"]["args"]


MISSING = "<missing>"


def train_py_defaults(train_py: str | Path | None = None) -> dict[str, Any]:
    """Argparse defaults of the released ``train.py``, read from its source (never executed).

    The checkpoint's saved args come from an older ``train.py``. Flags added later (e.g.
    ``sliding_window_inference``) are absent from it, but the released ``Dino_f`` reads them.
    ``store_true`` flags default to ``False``. Defaults that are not literals (e.g.
    ``type=parse_tuple`` strings) are passed through as written.
    """
    import ast
    import warnings

    source = Path(train_py or SUBMODULE / "train.py").read_text(encoding="utf-8")
    with warnings.catch_warnings():  # their help strings contain invalid escapes like "\l"
        warnings.simplefilter("ignore")
        tree = ast.parse(source)
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        long = next((f for f in flags if f.startswith("--")), None)
        if long is None:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "default" in kw:
            try:
                value = ast.literal_eval(kw["default"])
            except ValueError:
                value = None
        else:
            action = kw.get("action")
            value = False if isinstance(action, ast.Constant) and action.value == "store_true" else None
        defaults[long[2:].replace("-", "_")] = value
    return defaults


def inference_args(
    saved: Namespace,
    *,
    pca_ckpt: str | Path,
    img_size: tuple[int, int] | None = None,
    extra: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> tuple[Namespace, dict[str, tuple[Any, Any]]]:
    """Copy ``saved`` and apply the inference overrides. Returns ``(args, diff)``.

    Fields missing from ``saved`` (flags added to ``train.py`` after the checkpoint was
    trained) are first filled from ``defaults`` (default: ``train_py_defaults()``) and are
    reported in ``diff`` as ``("<missing>", value)``. ``img_size`` different from the trained
    size turns on ``high_res_adapt``, so the reference ``on_load_checkpoint`` bilinearly
    interpolates the positional-embedding tables to the new grid. ``extra`` adds arbitrary
    overrides (e.g. a DPT head). ``diff`` maps every changed field to ``(old, new)``.
    ``saved`` is never mutated.
    """
    args = copy.deepcopy(saved)
    diff: dict[str, tuple[Any, Any]] = {}
    for key, value in (train_py_defaults() if defaults is None else defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
            diff[key] = (MISSING, value)
    target_size = tuple(img_size) if img_size is not None else tuple(saved.img_size)
    overrides = {
        **_INFERENCE_OVERRIDES,
        "pca_ckpt": str(pca_ckpt),
        "img_size": target_size,
        "high_res_adapt": target_size != tuple(saved.img_size),
        **(extra or {}),
    }
    for key, new in overrides.items():
        old = getattr(saved, key, MISSING)
        if old != new:
            diff[key] = (old, new)
        setattr(args, key, new)
    return args, diff


def _purge_dinov2_modules() -> None:
    """Forget any imported DINOv2 hub code so a different commit can be loaded."""
    for name in list(sys.modules):
        if name == "hubconf" or name == "dinov2" or name.startswith("dinov2."):
            del sys.modules[name]


@contextmanager
def pinned_dinov2_hub(version: str = DINOV2_HUB_PINNED) -> Iterator[None]:
    """Route ``torch.hub.load('facebookresearch/dinov2', ...)`` to the pinned local code.

    Weights resolve from ``<hub_dir>/checkpoints`` (pre-downloaded), so no network is used.
    """
    code_dir = dinov2_code_dir(version)
    if not (code_dir / "hubconf.py").is_file():
        raise FileNotFoundError(f"{code_dir} has no hubconf.py; run projects/dinof_audit/download.py")
    original_load, original_dir = torch.hub.load, torch.hub.get_dir()

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).split(":")[0] == DINOV2_REPO:
            kwargs.update(source="local", trust_repo=None)
            kwargs.pop("force_reload", None)
            return original_load(str(code_dir), model, *args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    _purge_dinov2_modules()
    torch.hub.set_dir(str(dinov2_hub_dir()))
    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.set_dir(original_dir)


def load_dinov2_backbone(version: str = DINOV2_HUB_PINNED, device: str | torch.device = "cpu") -> nn.Module:
    """DINOv2-Reg ViT-B/14 built from the hub code at ``version`` with the official weights."""
    with pinned_dinov2_hub(version):
        model = torch.hub.load(DINOV2_REPO, "dinov2_vitb14_reg", pretrained=False)
    state = torch.load(dinov2_weights_path(), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def _import_dino_f():
    root = str(SUBMODULE)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.dino_f import Dino_f  # noqa: PLC0415  (submodule import, intentionally lazy)

    return Dino_f


def load_reference(
    ckpt: CkptLike,
    *,
    pca_ckpt: str | Path,
    img_size: tuple[int, int] | None = None,
    extra: dict[str, Any] | None = None,
    hub_version: str = DINOV2_HUB_PINNED,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate the reference ``Dino_f`` with the checkpoint's own args and load its weights.

    Mirrors ``LightningModule.load_from_checkpoint`` (``on_load_checkpoint``, then
    ``load_state_dict(strict=False)``) but keeps the missing/unexpected keys for the audit.
    The model is returned in ``eval()`` mode with gradients disabled.
    """
    ckpt = _load_ckpt(ckpt)
    args, diff = inference_args(checkpoint_args(ckpt), pca_ckpt=pca_ckpt, img_size=img_size, extra=extra)
    Dino_f = _import_dino_f()
    with pinned_dinov2_hub(hub_version):
        model = Dino_f(args)
    # on_load_checkpoint edits the state dict in place (positional-embedding interpolation).
    local = {**ckpt, "state_dict": dict(ckpt["state_dict"])}
    model.on_load_checkpoint(local)
    result = model.load_state_dict(local["state_dict"], strict=False)
    required = ("maskvit.", "embed.", "replace_vector")
    missing_required = [k for k in result.missing_keys if k.startswith(required)]
    if missing_required:
        raise RuntimeError(f"Checkpoint is missing world-model weights: {missing_required[:10]}")
    model.eval().requires_grad_(False).to(device)
    info = {
        "args_diff": {k: [repr(o), repr(n)] for k, (o, n) in diff.items()},
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "hub_version": hub_version,
    }
    return model, info


def center_crop_box(width: int, height: int, aspect: tuple[int, int]) -> tuple[int, int, int, int]:
    """Largest centered ``(left, top, crop_w, crop_h)`` box with ``crop_w / crop_h == aw / ah``."""
    aw, ah = aspect
    if width * ah >= height * aw:  # too wide: keep full height
        crop_h, crop_w = height, round(height * aw / ah)
    else:  # too tall: keep full width
        crop_w, crop_h = width, round(width * ah / aw)
    return (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h


def demo_frame_paths() -> list[Path]:
    """The 5 Cityscapes demo frames (stride 3: 4 context + 1 target), in temporal order."""
    paths = sorted(DEMO_DIR.glob("*_leftImg8bit.png"))
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected 5 demo frames in {DEMO_DIR}, found {len(paths)}")
    return paths


def load_demo_frames(
    img_size: tuple[int, int],
    crop_aspect: tuple[int, int] | list[tuple[int, int]] | None = None,
) -> Tensor:
    """``[1, 5, 3, H, W]`` demo clip with the reference eval transform (``src/data.py:165``).

    ``Resize(img_size)`` (exact size, no aspect preservation) → ``ToTensor`` → ImageNet
    ``Normalize``, optionally preceded by centered crops: ``crop_aspect = (w, h)`` or a list
    of aspects applied in order (e.g. ``[(5, 4), (2, 1)]`` = FLIR frame, then cropped to 2:1).
    """
    import torchvision.transforms as T
    from PIL import Image

    if crop_aspect is None:
        aspects: list[tuple[int, int]] = []
    elif isinstance(crop_aspect[0], int):
        aspects = [tuple(crop_aspect)]  # type: ignore[list-item]
    else:
        aspects = [tuple(a) for a in crop_aspect]  # type: ignore[union-attr]
    transform = T.Compose([T.Resize(tuple(img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    frames = []
    for p in demo_frame_paths():
        img = Image.open(p).convert("RGB")
        for aspect in aspects:
            left, top, cw, ch = center_crop_box(*img.size, aspect)
            img = img.crop((left, top, left + cw, top + ch))
        frames.append(transform(img))
    return torch.stack(frames).unsqueeze(0)


def one_step(model, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Reference 1-step prediction in PCA space: ``x [B,5,h,w,1152]`` → (pred last frame, loss).

    Exactly ``Dino_f.sample`` minus pre/post-processing (``src/dino_f.py:441-447``): the last
    frame is fully masked and its SmoothL1 against ``x[:, -1]`` is the reference loss.
    """
    masked, mask = model.get_mask_tokens(x, mode="full_mask", mask_frames=1)
    loss, final = model.forward(x, masked, mask.to(x.device))
    return final[:, -1], float(loss)


def pca_metrics(pred: torch.Tensor, gt: torch.Tensor, beta: float) -> dict:
    """SmoothL1(beta), MSE and mean per-token cosine between ``[..., C]`` tensors."""
    pred, gt = pred.float(), gt.float()
    return {
        "smoothl1": float(F.smooth_l1_loss(pred, gt, beta=beta)),
        "mse": float(F.mse_loss(pred, gt)),
        "cosine": float(F.cosine_similarity(pred, gt, dim=-1).mean()),
    }
