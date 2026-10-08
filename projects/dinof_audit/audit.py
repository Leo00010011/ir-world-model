"""F1 audit run: execute the unmodified DINO-Foresight reference code and record the evidence.

Offline (run download.py first). GPU recommended:

    slurm/submit.sh ctcomp3 projects/dinof_audit/audit.py cluster.timeout_min=60 cluster.mem_gb=32
    uv run python projects/dinof_audit/audit.py device=cpu       # works too, slower

Writes <run_dir>/audit_results.json and <run_dir>/figures/*.png, and logs both to W&B.
Sections follow spec/2026-10-02-dino-foresight-audit/requirements.md FR7.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from irwm.features.pca import PCABasis  # noqa: E402
from irwm.utils.paths import load_env  # noqa: E402
from irwm.utils.tracking import git_info, make_wandb_logger  # noqa: E402
from irwm.worldmodel import assets as A  # noqa: E402
from irwm.worldmodel import reference as R  # noqa: E402

load_env()

LAYERS = [2, 5, 8, 11]
CITYSCAPES_PALETTE = torch.tensor([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=torch.uint8)


# ----------------------------------------------------------------------------- helpers
def _json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    return repr(x)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


class Ctx:
    """Shared state between sections (loaded lazily, reused)."""

    def __init__(self, cfg: DictConfig, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.fig_dir = run_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if cfg.device == "auto" else cfg.device))
        self.ckpt_path = A.dinof_dir() / A.DINOF_HIGHRES
        self.pca_path = A.dinof_dir() / A.DINOF_PCA
        self._ckpt: dict | None = None
        self._model = None
        self.load_info: dict = {}
        self.ckpt_meta: dict = {}

    @property
    def ckpt(self) -> dict:
        if self._ckpt is None:
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.ckpt_meta = {
                "top_level_keys": sorted(ck.keys()),
                "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
                "pytorch_lightning_version": ck.get("pytorch-lightning_version"),
            }
            for heavy in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
                ck.pop(heavy, None)  # free RAM; only state_dict + hparams are needed
            self._ckpt = ck
        return self._ckpt

    @property
    def model(self):
        if self._model is None:
            self._model, self.load_info = R.load_reference(self.ckpt, pca_ckpt=self.pca_path, device=self.device)
        return self._model

    def frames(self, img_size=(448, 896), crop=None) -> torch.Tensor:
        return R.load_demo_frames(img_size, crop).to(self.device)


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


def forecast_eval(model, frames: torch.Tensor) -> dict:
    """1-step prediction vs truth and vs copy-last, in PCA space."""
    beta = model.args.beta_smoothl1
    with torch.no_grad():
        x = model.preprocess(frames)
        pred, loss = one_step(model, x.clone())
    gt, last = x[:, -1], x[:, -2]
    m, c = pca_metrics(pred, gt, beta), pca_metrics(last, gt, beta)
    return {"model": m, "copy_last": c, "reference_loss": loss,
            "ratio_smoothl1": m["smoothl1"] / c["smoothl1"], "grid": list(x.shape[2:4])}


# ----------------------------------------------------------------------------- sections
def section_artefacts(ctx: Ctx) -> dict:
    args = R.checkpoint_args(ctx.ckpt)
    model = ctx.model
    sd_groups = Counter(k.split(".")[0] for k in ctx.ckpt["state_dict"])
    params = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    return {
        "verify_manifest": {"dinof_dir": A.verify_manifest(A.dinof_dir()),
                            "dinov2_hub_dir": A.verify_manifest(A.dinov2_hub_dir())},
        "checkpoint": ctx.ckpt_meta,
        "saved_args": {k: _json(v) for k, v in sorted(vars(args).items())},
        "state_dict_groups": dict(sd_groups),
        "params_per_module": params,
        "params_world_model": sum(v for k, v in params.items() if k != "dino_v2"),
        "load_info": ctx.load_info,
    }


def section_weights(ctx: Ctx) -> dict:
    sd = ctx.ckpt["state_dict"]
    official = torch.load(A.dinov2_weights_path(), map_location="cpu", weights_only=True)
    ck_dino = {k[len("dino_v2."):]: v for k, v in sd.items() if k.startswith("dino_v2.")}
    common = sorted(set(ck_dino) & set(official))
    dino = {
        "n_ckpt": len(ck_dino), "n_official": len(official), "n_common": len(common),
        "only_in_ckpt": sorted(set(ck_dino) - set(official))[:20],
        "only_in_official": sorted(set(official) - set(ck_dino))[:20],
        "max_abs_diff": max((maxdiff(ck_dino[k], official[k]) for k in common), default=None),
    }
    basis = PCABasis.load(A.dinof_dir() / A.PCA_EXPORT)
    pca_ck = {"pca_mean": "pca_mean", "components": "pca_components", "mean": "mean", "std": "std"}
    pca = {ours: maxdiff(getattr(basis, ours), sd[theirs]) for ours, theirs in pca_ck.items() if theirs in sd}
    pca["dtypes_in_ckpt"] = {theirs: str(sd[theirs].dtype) for theirs in pca_ck.values() if theirs in sd}
    model = ctx.model
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 2048, 3072, generator=g).to(ctx.device)
    z = torch.randn(1, 1, 32, 64, 1152, generator=g).to(ctx.device)
    basis = basis.to(ctx.device)
    with torch.no_grad():
        equiv = {"transform_max_abs_diff": maxdiff(basis.transform(x), model.pca_trantxsform(x)),
                 "inverse_max_abs_diff": maxdiff(basis.inverse(z), model.pca_inverse_transform(z))}
    return {"dinov2_ckpt_vs_official": dino, "pca_ckpt_vs_export": pca, "pcabasis_vs_reference": equiv,
            "pcabasis_sha256": basis.sha256}


def section_recipe(ctx: Ctx) -> dict:
    model = ctx.model
    dino = model.dino_v2
    flat = ctx.frames().flatten(end_dim=1)  # [5, 3, 448, 896]
    hooked: dict[int, torch.Tensor] = {}
    blocks = dino.blocks
    if getattr(dino, "chunked_blocks", False):  # BlockChunk lists: flatten to real blocks
        blocks = [b for chunk in dino.blocks for b in chunk if not isinstance(b, torch.nn.Identity)]
    handles = [blocks[i].register_forward_hook(lambda m, inp, out, i=i: hooked.__setitem__(i, out)) for i in LAYERS]
    with torch.no_grad():
        feats = model.extract_features(flat)  # reference path: get_intermediate_layers(n=[2,5,8,11])
        raw = dino.get_intermediate_layers(flat, n=LAYERS, reshape=False, norm=False)
    for h in handles:
        h.remove()
    n_skip = 1 + dino.num_register_tokens
    manual = torch.cat([dino.norm(hooked[i])[:, n_skip:] for i in LAYERS], dim=-1)
    manual_nonorm = torch.cat([hooked[i][:, n_skip:] for i in LAYERS], dim=-1)
    with torch.no_grad():
        z = (feats - model.mean) / model.std
        zc = z - model.pca_mean
        coeffs = zc @ model.pca_components.T
        recon = coeffs @ model.pca_components
        rec_full = model.pca_inverse_transform(coeffs.reshape(1, 5, 32, 64, -1)).reshape_as(feats)
    return {
        "tokens_total_per_image": int(hooked[LAYERS[0]].shape[1]),
        "num_register_tokens": int(dino.num_register_tokens),
        "patch_tokens": int(feats.shape[1]),
        "feature_dim": int(feats.shape[2]),
        "embed_dim_per_layer": int(dino.embed_dim),
        "intermediate_equals_hooks_plus_norm": maxdiff(feats, manual),
        "intermediate_norm_false_equals_raw_hooks": maxdiff(raw[0], manual_nonorm[..., :dino.embed_dim]),
        "effect_of_final_norm_max_abs": maxdiff(feats, manual_nonorm),
        "zscore": {"mean_abs_of_dim_means": float(z.mean(dim=(0, 1)).abs().mean()),
                   "mean_of_dim_stds": float(z.std(dim=(0, 1)).mean())},
        "pca_explained_variance_demo": float(1 - (zc - recon).pow(2).sum() / zc.pow(2).sum()),
        "pca_relative_recon_error_raw": float((feats - rec_full).pow(2).sum() / (feats - feats.mean(dim=(0, 1))).pow(2).sum()),
        "hub_version_used": ctx.load_info.get("hub_version"),
    }


def section_rollout(ctx: Ctx) -> dict:
    model = ctx.model
    beta = model.args.beta_smoothl1
    frames = ctx.frames()
    out: dict[str, Any] = {"forecast_1step": forecast_eval(model, frames)}
    with torch.no_grad():
        x = model.preprocess(frames)
        pred_full, loss_sample = model.sample(frames)  # reference public path
        pred1, _ = one_step(model, x.clone())
        out["shapes"] = {"pca_input": list(x.shape), "sample_output_full": list(pred_full.shape)}
        out["sample_equals_one_step_full"] = maxdiff(pred_full[:, -1], model.postprocess(pred1.unsqueeze(1))[:, 0])

        # 3-step: reference sample_unroll vs manual PCA-space loop
        gt_full = model.extract_features(frames[:, -1]).reshape(1, 32, 64, -1)
        unroll_full, _ = model.sample_unroll(frames, gt_full, unroll_steps=3)
        xm = model.preprocess(frames)
        steps = []
        for _ in range(3):
            p, _ = one_step(model, xm.clone())
            steps.append(p)
            xm[:, -1] = p
            xm = torch.cat((xm[:, 1:], xm[:, -1:]), dim=1)
        out["unroll3_equals_manual_pca_loop"] = maxdiff(unroll_full, model.postprocess(xm))
        out["unroll3_output_shape"] = list(unroll_full.shape)
        out["unroll_step_cosine_to_previous_step"] = [
            float(F.cosine_similarity(steps[i], steps[i - 1], dim=-1).mean()) for i in range(1, 3)]

        # Masked target slot must not influence the prediction
        f0, fn = frames.clone(), frames.clone()
        f0[:, -1] = 0
        fn[:, -1] = torch.randn_like(fn[:, -1])
        p0, _ = one_step(model, model.preprocess(f0))
        pn, _ = one_step(model, model.preprocess(fn))
        out["masked_slot_independence"] = {"zeros": maxdiff(p0, pred1), "noise": maxdiff(pn, pred1)}

        # Determinism
        p_again, _ = one_step(model, model.preprocess(frames))
        out["determinism_max_abs_diff"] = maxdiff(p_again, pred1)

        # Precision
        if ctx.cfg.precision_check and ctx.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                xh = model.preprocess(frames)
                ph, _ = one_step(model, xh.clone())
            gt = x[:, -1]
            out["precision"] = {
                "fp16_autocast_vs_fp32_max_abs": maxdiff(ph, pred1),
                "fp16_autocast_vs_fp32_mean_abs": float((ph.float() - pred1).abs().mean()),
                "smoothl1_fp32": pca_metrics(pred1, gt, beta)["smoothl1"],
                "smoothl1_fp16_autocast": pca_metrics(ph, gt, beta)["smoothl1"],
            }

        # Dummy PCA-space inputs, batch 2
        g = torch.Generator().manual_seed(0)
        xd = torch.randn(2, 5, 32, 64, 1152, generator=g).to(ctx.device) * x.std()
        pd, _ = one_step(model, xd)
        out["dummy_batch2"] = {"shape": list(pd.shape), "finite": bool(torch.isfinite(pd).all())}

        # Timing
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
            ts = []
            for _ in range(ctx.cfg.timing_repeats):
                t0 = time.perf_counter()
                one_step(model, x.clone())
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            out["timing_1step_forward_s"] = {"median": sorted(ts)[len(ts) // 2], "gpu": torch.cuda.get_device_name(0)}
    return out


def section_hub(ctx: Ctx) -> dict:
    model = ctx.model
    flat = ctx.frames().flatten(end_dim=1)
    feats, res = {}, {}
    for name in ctx.cfg.hub_versions:
        backbone = R.load_dinov2_backbone(name, device=ctx.device)
        with torch.no_grad():
            f = torch.cat(backbone.get_intermediate_layers(flat, n=LAYERS, reshape=False), dim=-1)
            x = model.pca_transform(f).reshape(1, 5, 32, 64, -1)
            _, loss = one_step(model, x)
            z = (f - model.mean) / model.std
        feats[name] = f
        res[name] = {"smoothl1_1step": loss, "zscore_mean_abs_of_dim_means": float(z.mean(dim=(0, 1)).abs().mean()),
                     "zscore_mean_of_dim_stds": float(z.std(dim=(0, 1)).mean())}
        del backbone
    ref = "dinof_release" if "dinof_release" in feats else next(iter(feats))
    for name in feats:
        res[name]["feature_max_abs_diff_vs_" + ref] = maxdiff(feats[name], feats[ref])
    best = min(res, key=lambda n: res[n]["smoothl1_1step"])
    tie = all(res[n]["feature_max_abs_diff_vs_" + ref] < 1e-5 for n in res)
    chosen = "dinof_release" if tie and "dinof_release" in res else best
    return {"versions": res, "commits": {n: A.DINOV2_HUB_COMMITS[n] for n in res}, "lowest_loss": best,
            "all_identical": tie, "recommended_pin": chosen, "currently_pinned": A.DINOV2_HUB_PINNED}


def section_r5(ctx: Ctx) -> dict:
    cfg = ctx.cfg.r5
    aspect = tuple(cfg.crop_aspect)
    native = tuple(cfg.native_size)
    model = ctx.model
    res = {
        "reference_2to1_uncropped": forecast_eval(model, ctx.frames()),
        "a_stretch_448x896": forecast_eval(model, ctx.frames((448, 896), aspect)),
        "b_crop2to1_448x896": forecast_eval(model, ctx.frames((448, 896), [aspect, (2, 1)])),
    }
    model_c, info_c = R.load_reference(ctx.ckpt, pca_ckpt=ctx.pca_path, img_size=native, device=ctx.device)
    res["c_native_%dx%d" % native] = forecast_eval(model_c, ctx.frames(native, aspect))
    res["c_load_info"] = {"args_diff": info_c["args_diff"],
                          "pos_emb_shapes": {k: list(v.shape) for k, v in model_c.maskvit.pos_embd.emb.items()}}
    del model_c
    policies = {k: v["ratio_smoothl1"] for k, v in res.items() if k[:2] in ("a_", "b_", "c_") and "ratio_smoothl1" in v}
    best = min(policies, key=policies.get)
    stretch = next(k for k in policies if k.startswith("a_"))
    decision = stretch if policies[stretch] <= policies[best] * (1 + cfg.tie_tolerance) else best
    res["ratios"] = policies
    res["decision"] = {"best": best, "chosen": decision, "rule": f"lowest ratio; stretch if within {cfg.tie_tolerance:.0%}"}
    return res


def section_segm(ctx: Ctx) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dpt import DPTHead  # submodule root is on sys.path after load_reference

    model = ctx.model
    head_path = A.dinof_dir() / "head_segm_pca1152.ckpt"
    hck = torch.load(head_path, map_location="cpu", weights_only=False)
    # Hyper-parameters as in the README evaluation command (dpt_out_channels 128,256,512,512, --use_bn, nfeats 256).
    head = DPTHead(nclass=19, in_channels=model.feature_dim, features=256, use_bn=True,
                   out_channels=[128, 256, 512, 512], use_clstoken=False)
    sd = {k.replace("head.", ""): v for k, v in hck["state_dict"].items()}
    load = head.load_state_dict(sd, strict=False)
    head = head.eval().to(ctx.device)

    frames = ctx.frames()
    with torch.no_grad():
        x = model.preprocess(frames)
        pred, _ = one_step(model, x.clone())
        feats = {
            "oracle": model.extract_features(frames[:, -1]).reshape(1, 32, 64, -1),
            "oracle_pca_roundtrip": model.postprocess(x[:, -1:])[:, 0],
            "copy_last": model.postprocess(x[:, -2:-1])[:, 0],
            "predicted": model.postprocess(pred.unsqueeze(1))[:, 0],
        }
        segs = {}
        for k, f in feats.items():
            parts = [f[..., i * 768:(i + 1) * 768].reshape(1, 32 * 64, 768) for i in range(4)]
            logits = F.interpolate(head(parts, 32, 64), size=(1024, 2048), mode="bilinear", align_corners=False)
            segs[k] = logits.argmax(1)[0].cpu()
    agree = {k: float((segs[k] == segs["oracle"]).float().mean()) for k in segs if k != "oracle"}

    from PIL import Image
    rgb = Image.open(R.demo_frame_paths()[-1]).convert("RGB")
    fig, axes = plt.subplots(1, 5, figsize=(25, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("target frame (t+3)")
    for ax, (k, s) in zip(axes[1:], segs.items()):
        ax.imshow(CITYSCAPES_PALETTE[s].numpy())
        ax.set_title(f"{k}" + (f"  (agree {agree[k]:.1%})" if k in agree else ""))
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path = ctx.fig_dir / "segm_demo.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return {"pixel_agreement_with_oracle": agree, "head_ckpt_keys": sorted(hck.keys()),
            "head_missing_keys": load.missing_keys, "head_unexpected_keys": load.unexpected_keys,
            "figure": str(path)}


SECTIONS: dict[str, Callable[[Ctx], dict]] = {
    "artefacts": section_artefacts, "weights": section_weights, "recipe": section_recipe,
    "rollout": section_rollout, "hub": section_hub, "r5": section_r5, "segm": section_segm,
}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


@hydra.main(version_base="1.3", config_path="conf", config_name="audit")
def main(cfg: DictConfig) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ctx = Ctx(cfg, run_dir)
    results: dict[str, Any] = {
        "git": git_info(), "config": OmegaConf.to_container(cfg, resolve=True),
        "hf_revision": A.DINOF_HF_REVISION, "dinov2_hub_pinned": {A.DINOV2_HUB_PINNED: A.DINOV2_HUB_COMMITS[A.DINOV2_HUB_PINNED]},
        "device": str(ctx.device), "torch": torch.__version__, "errors": {},
    }
    out_path = run_dir / "audit_results.json"
    for name in cfg.sections:
        print(f"\n===== section {name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            results[name] = SECTIONS[name](ctx)
        except Exception as e:  # keep going: a failed section is itself an audit finding
            import traceback

            results["errors"][name] = traceback.format_exc()
            print(results["errors"][name], flush=True)
        results.setdefault("section_seconds", {})[name] = round(time.perf_counter() - t0, 2)
        out_path.write_text(json.dumps(_json(results), indent=2))  # incremental, survives a crash
        print(json.dumps(_json(results.get(name, {})), indent=2)[:4000], flush=True)

    try:
        import wandb

        logger = make_wandb_logger(cfg, save_dir=run_dir)
        logger.experiment.log(_flatten({k: v for k, v in results.items() if k in SECTIONS}))
        for png in sorted(ctx.fig_dir.glob("*.png")):
            logger.experiment.log({f"figures/{png.stem}": wandb.Image(str(png))})
        logger.experiment.save(str(out_path), base_path=str(run_dir))
        logger.experiment.finish()
    except Exception as e:  # W&B problems must not lose the audit results
        print(f"[audit] W&B logging failed: {e!r}")
    print(f"\n[audit] results: {out_path}")
    if results["errors"]:
        raise SystemExit(f"[audit] sections with errors: {sorted(results['errors'])}")


if __name__ == "__main__":
    main()
