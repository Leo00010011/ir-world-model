"""Feature stem: takes the place of ViTDet's ViT-B in front of detrex's simple feature pyramid.

Spec: spec/2026-10-23-dino-detr-head-port (FR4, deviations D1 and D5). Per token, LayerNorm over
the C input channels (eps 1e-6, as the ViT's ``norm_layer``, ``dino_vitdet.py:26``), then a linear
projection C → 768, so everything from the pyramid on has ViTDet-B's shapes for C = 1152 or 3072.

Backbone ablation (F17, TechStack §3.1b): with ``proj_channels`` set, a learned
``in_proj = Linear(C → proj_channels)`` comes first (``dino3072``: 3072 → 1152, in place of the
fixed PCA), then LayerNorm(proj_channels) and the stem's own Linear. The LayerNorm between the two
linear maps keeps them from collapsing into one. Without ``proj_channels`` (the default) no
submodule is added, so the state_dict of existing heads is unchanged.
"""

from __future__ import annotations

from detectron2.modeling import Backbone
from torch import Tensor, nn


class FeatureStem(Backbone):
    """``[B, C, H', W']`` frozen encoder features → ``{out_feature: [B, embed_dim, H', W']}``."""

    def __init__(self, in_channels: int = 1152, embed_dim: int = 768, out_feature: str = "last_feat",
                 proj_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.proj_channels = proj_channels
        width = in_channels
        if proj_channels is not None:  # registered only when used: default keys stay {norm, proj}.*
            self.in_proj = nn.Linear(in_channels, proj_channels)
            width = proj_channels
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, embed_dim)
        self._out_features = [out_feature]
        self._out_feature_channels = {out_feature: embed_dim}
        # Nominal stride (D5): SimpleFeaturePyramid names its levels from stride / scale, and 16
        # gives p3-p6 as detrex's neck expects. The real token pitch is 14 px; DINO never uses it.
        self._out_feature_strides = {out_feature: 16}

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"FeatureStem expects a 4-D [B, {self.in_channels}, H', W'] tensor, "
                             f"got {list(x.shape)}")
        y = x.permute(0, 2, 3, 1)
        if self.proj_channels is not None:
            y = self.in_proj(y)
        y = self.proj(self.norm(y))
        return {self._out_features[0]: y.permute(0, 3, 1, 2).contiguous()}
    # output_shape() is detectron2 Backbone's, built from the _out_feature_* dicts above.
