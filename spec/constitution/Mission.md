# Mission

## One-line summary

Show that the features predicted by a latent-space world model don't keep all the representational power of the real encoder features. If they don't, future-prediction error and representation quality are two separate things that can be measured separately. Then find unsupervised ways to close that gap.

## Change log

- **2026-11-13: aligned FLIR v1 and a backbone ablation added (user).** Two detector-level additions, both outside the world-model path:
  - **Aligned FLIR v1** (5,142 pixel-aligned RGB/thermal pairs, one shared annotation per pair) is a second detection dataset. Heads are trained, selected and tested on it, with person + car as the only classes. Because both modalities share the same frames and the same boxes, it tests H1 without the annotation asymmetry of FLIR v2 (R10). Aligned heads are **never scored on FLIR v2** (or v2 heads on aligned): the two datasets label objects differently, so cross-dataset scores would count many spurious false positives. FLIR v2 video test stays the only world-model test set.
  - **Backbone ablation (B)** on aligned v1: DINOv2-Reg + PCA (the current pipeline), DINOv2-Reg without PCA (3072-d) and MAE ViT-B/16. It asks how good the DINO+PCA features the world model reads are for detection.
  - See Roadmap Phase 2b (F16–F19).
- **2026-10-09: label space fixed to 6 FLIR classes (F2).** Mission originally planned 3 coarse classes (vehicle, pedestrian, cyclist). After a class-imbalance study of the as-downloaded annotations, the label space is the 6 most frequent FLIR classes: person, bike, car, motor, light, sign. There is no coarse mapping. Boxes of every other class (truck, bus, other vehicle, …) are ignore regions: neither targets nor background. Size buckets (APS/M/L) use areas normalised to a 640×512 frame, so RGB and thermal buckets cover the same objects. See `spec/2026-10-08-flir-readers-class-mapping/`.
- **2026-10-01: world model switched from FlowWM to DINO-Foresight.** FlowWM ([Porcher et al., 2026, arXiv:2606.29059](https://arxiv.org/abs/2606.29059)) has not released its weights, and we cannot retrain it. DINO-Foresight ([Karypidis et al., NeurIPS 2025, arXiv:2412.11673](https://arxiv.org/abs/2412.11673)) follows the same recipe: forecast in the feature space of a frozen DINO encoder, then decode with frozen task heads. Its checkpoints are public. The research questions are unchanged. What changes is the encoder (DINOv2 instead of DINOv3), the feature space (PCA-reduced), the temporal protocol, and the fact that the model is deterministic. If FlowWM weights appear later, it can be added as a second world model behind the same adapter API.

## The problem

Latent world models such as DINO-Foresight forecast future frames in the feature space of a frozen vision foundation model (DINOv2). They are judged by running frozen downstream heads on the predicted features and comparing against an *oracle*: the same heads on the real future features. The benchmarks report a single number, and that number mixes two kinds of error:

1. **Forecasting error**: the model predicts the wrong future (objects in the wrong place, missing, or hallucinated).
2. **Representation error**: the predicted features are off the encoder's feature manifold. They are blurrier, noisier and less useful to the heads than real features, even where the predicted future is "right". DINO-Foresight is a deterministic regressor trained with SmoothL1, so its predictions tend toward the conditional mean, which makes this error especially likely.

The hypothesis is that (2) is large and has been overlooked. A **domain shift** makes it visible: infrared (IR) imagery. DINO-Foresight was trained only on Cityscapes RGB, and its PCA basis was fit on Cityscapes RGB too.

## Core hypothesis and how we test it

**H1 (encoder level): DINOv2 features of IR images are better for detection than DINOv2 features of RGB images on FLIR ADAS.**
We train one DINO-DETR head on frozen DINOv2 features for each modality and evaluate each on its own modality. We expect the IR detector to match or beat the RGB detector, because thermal imaging helps at night, in glare and in bad weather.
*Paired check (aligned FLIR v1):* on FLIR v2 the modalities have different annotation densities (Roadmap R10). We therefore repeat H1 on aligned FLIR v1, where RGB and IR are the same frames with the same boxes. There, any IR–RGB gap comes from the image content alone.

**B (backbone control, aligned FLIR v1): how good is the world model's feature space for detection?**
The world model reads and writes DINOv2-Reg features reduced by a Cityscapes-RGB PCA. We train the same DINO-DETR head per modality on three frozen backbones:
- **DINOv2-Reg + PCA** (1152-d): the feature space of the world model
- **DINOv2-Reg without PCA**: the 3072-d concat goes through a learned linear layer to 1152-d. It measures what the fixed Cityscapes PCA costs each modality, compared with a learned projection of the same size (R7)
- **MAE ViT-B/16, fine-tuned** (user, 2026-11-13): detrex's **stock DINO-ViTDet** (`dino_vitdet`, ViT-B backbone with window + global attention and a simple feature pyramid), initialised from MAE and **trained end to end** with ViTDet's layer-wise lr decay 0.7. It follows the published recipe and is a supervised reference that measures how far the frozen DINOv2 features are from a backbone adapted to the target data.

Both DINOv2 variants stay frozen and hand the same 1152-d input to an identical head. The MAE variant differs in both architecture and training (fine-tuned), so it reads as a reference point rather than a controlled frozen-feature comparison.

Only DINOv2 + PCA can feed the world model, so B is an encoder-level result. It qualifies H1–H3 and does not replace them.

**H2 (world-model level): the world model does not carry that advantage over.**
DINO-Foresight runs **zero-shot** (no retraining) on FLIR video. We feed it 4 context frames, roll out future features, and apply the frozen per-modality detectors to the predictions. Two outcomes are possible:

- *Null outcome*: the IR ≥ RGB ordering from H1 survives in the predicted features. The world model keeps representation quality.
- *Hypothesised outcome*: the predicted IR features detect **worse than predicted RGB**, even though real IR features detect better. The IR benefit is lost at prediction time. So the predictions are not only wrong about the future; their representations are also degraded.

**H3 (decoupling):** representation error can be measured on its own, separately from forecasting error. DINO-Foresight's pipeline has three places where it can enter, and each can be measured:

- *PCA truncation*: real features projected to 1152 dimensions and back. The Cityscapes-RGB basis may lose more on IR.
- *Single-step prediction*: one step ahead (≈0.17 s), where the future is almost determined.
- *Autoregressive drift*: the gap growing over rollout steps.

Other measures include distance of predicted features to the real-feature manifold. The exact analyses are defined in Phase 3 of the Roadmap.

**H4 (adaptation, later phase):** unsupervised techniques can reduce the representation error without labels in the target domain, and the gains should carry over to other domains. Candidates include test-time adaptation, feature-manifold projection, re-fitting PCA on the target domain, and self-supervised fine-tuning of DINO-Foresight on unlabeled IR video. Its own masked-feature loss needs no labels.

## Dataset

**FLIR ADAS v2** (Teledyne), with paired thermal and RGB cameras and COCO-format annotations (80 COCO-style categories declared, 16 with boxes; we use 6, see Evaluation protocol). The user obtains the data; the repo only reads it.

| Split | Temporal? | Role in this project |
|---|---|---|
| `images_{thermal,rgb}_train` | No (sparse frames) | Train the DINO-DETR heads |
| `images_{thermal,rgb}_val` | No (sparse frames) | Choose detector checkpoints and hyperparameters |
| `video_{thermal,rgb}_test` | Yes (30 FPS sequences) | **The single test set**: oracle detection and world-model evaluation |

The oracle (detector on real features) and the world model (detector on predicted features) are scored on the **same video frames**. That makes the "oracle minus prediction" gap a direct comparison.

**Aligned FLIR v1** (Zhang et al., 2020; added 2026-11-13) is a second, detection-only dataset under `$IRWM_DATA_ROOT/aligned/align/`. It holds 5,142 RGB/thermal pairs at 640×512 that are pixel-aligned, with **one VOC annotation per pair** shared by both modalities. Its classes are car, person, bicycle and dog. It has no video, so it cannot be used for the world model.

| Split | Size | Role in this project |
|---|---|---|
| `align_train.txt` minus a held-out block | ≈ 3.7k pairs | Train the aligned heads |
| held-out block of `align_train.txt` (contiguous ids at its end, ≈ 10 %) | ≈ 0.4k pairs | Choose checkpoints (aligned val) |
| `align_validation.txt` (ids 08864–10227, disjoint in time from train) | 1,013 pairs | **Aligned test set** (the literature's convention) |

## Overall pipeline

```
FLIR ADAS v2 (raw, user-provided)
        │
        ▼
[1] Data modules ── read images/videos + COCO labels, build clips
        │                (4 context frames + K future steps, step ≈ 0.17 s)
        ▼
[2] Feature cache ─ frozen DINOv2-Reg ViT-B/14, blocks 3/6/9/12 concatenated (3072-d)
        │                → DINO-Foresight's PCA (1152-d), fp16
        │                (computed once, reused by every downstream stage)
        ├──────────────────────────────┐
        ▼                              ▼
[3] DINO-DETR heads            [5] DINO-Foresight zero-shot rollouts
    (train on cached features,     (4 context features → K autoregressive steps,
     one per modality)              cached as predicted features)
        │                              │
        ▼                              ▼
[4] Oracle eval (real features) ── [6] Prediction eval (predicted features)
        │                              │
        └──────────────┬───────────────┘
                       ▼
          [7] Analysis: H1, H2, H3 tables and figures
                       │
                       ▼
          [8] Unsupervised adaptation experiments (H4)

Aligned FLIR v1 (side branch, detector level only: H1 paired check + B)
  readers ─► {DINOv2+PCA, DINOv2 3072-d, MAE} × {RGB, IR} heads ─► aligned test 2×2 per backbone
```

## Evaluation protocol

- **Primary metric:** COCO **AP** and **AP50** over the **6 FLIR classes person, bike, car, motor, light, sign** (`irwm.data.classes`, fixed by F2). Boxes of every other FLIR class are **ignore regions** (exported as crowd GT, so detections on them are neither TP nor FP). This mirrors DINO-Foresight's instance-level metrics (AP, AP50).
- **Secondary metrics:** APS/M/L with areas **normalised to a 640×512 frame** (APL was the FlowWM paper's metric, so we keep it for comparison), plus per-class AP. Per-class comparisons across modalities must account for the annotation asymmetry in the test split (Roadmap R10).
- **Horizons:** **short-term = 1 step** (≈0.17 s) and **mid-term = 3 autoregressive steps** (≈0.5 s), as in DINO-Foresight. For the horizon curve we extend to 6 steps (≈1.0 s).
- **Determinism:** DINO-Foresight is deterministic, so there is a single prediction per clip and no best-of-N.
- **Baselines, all scored on the same frames:**
  - oracle: real future features
  - **PCA round-trip oracle**: real features passed through PCA and back
  - **copy-last**: the last context frame's real features
- **Comparison:** each detector is scored on its own modality's full test split with that modality's own annotations. RGB and IR frames are not paired one-to-one.
- **Aligned FLIR v1 protocol (2026-11-13):** the label space is **person and car only**, with a **2-way head** (person = 0, car = 1; user decision 2026-11-13). **Bicycle and dog are background**: they are dropped from targets and GT, so a detection on them is a false positive (user decision 2026-11-13; this departs from the F2 ignore rule). The metric is AP/AP50 over person + car on `align_validation`, plus per-class AP. Every head is scored on both modalities of the aligned test set (one 2×2 per backbone; the boxes are identical, so cross-modal cells are fair). Aligned and v2 heads are never scored across datasets, because the annotation conventions differ. Comparisons with v2 use v2's person + car AP and are made with caution, because the datasets, test sets and splits all differ.
- **Sanity check:** on FLIR **RGB**, the model must beat copy-last (in feature distance at least, ideally in AP), stay below the oracle, and degrade with horizon. We do not reproduce the Cityscapes results.

## Desired behaviours (applies to all code)

- **Compute once, reuse forever.** Anything expensive and deterministic (encoder features, world-model rollouts) is cached to disk with a documented layout, and is regenerated only when its config changes.
- **Frozen means frozen.** DINOv2 and DINO-Foresight weights (including the PCA) are never updated outside Phase 4. Code must make this explicit (`requires_grad_(False)`, `eval()`).
- **Paper fidelity first.** Where DINO-Foresight's code or paper fixes a choice (encoder, layers, PCA, resolution, context length, step size, rollout scheme), we copy it. Any deviation is written into the spec that introduces it. Where the code and the paper disagree, **the code wins**, because the checkpoint was trained with the code.
- **Every number is reproducible.** Every result traces back to a Hydra config, a git commit and a W&B run.
- **Cluster-first.** Everything runs non-interactively under SLURM, survives preemption through checkpoints and resume, and never hard-codes a filesystem path.
- **Features are small and independent.** Each Roadmap feature can be built and validated in one session.
