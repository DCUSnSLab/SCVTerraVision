"""DINOv3 ViT-B/16 backbone wrapper for detection heads.

Wraps `transformers.AutoModel.from_pretrained(...)` around the gated
`facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint and returns a dense
(B, C, H_patch, W_patch) feature map. That's the layout DETR-style heads
(Phase 1-2b) expect — (B, C, H, W), not the raw (B, N_tokens, C) HF output.

Design decisions:
  - torch and transformers are imported **inside** `load()` / `forward()` so
    this module imports cleanly in environments without them (Phase 1-2a
    dev machines). Config construction and `patch_grid_shape(...)` work
    without any ML deps — enough to ship Hydra configs and round-trip them.
  - DINOv3 prepends 1 CLS + K register tokens (K=4 in the current HF
    implementation). Rather than hardcoding K, we slice the last
    `H_patch * W_patch` tokens from `last_hidden_state`. That's robust to
    register-count changes across checkpoints.
  - `freeze=True` puts the backbone in eval() and disables grads. Phase
    1-2b's training loop calls `unfreeze_after(epoch, schedule)` to flip
    this on a schedule.

License: DINOv3 is gated (own DINOv3 License, not Apache-2.0). The HF
account must accept the license once, and HF_TOKEN must be in env at
load time. See docs/decisions/20260422_dinov3-backbone.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Torch / transformers are intentionally NOT imported at module scope. See
# docstring above. All methods that need them do a local import and fail
# with a helpful message if they're absent.

DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_PATCH_SIZE = 16
DEFAULT_HIDDEN_DIM = 768  # ViT-B
_SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class DinoV3BackboneConfig:
    """Serializable config for the DINOv3 backbone wrapper.

    Kept frozen + dataclass so Hydra/OmegaConf can round-trip it as a
    structured config without surprises (no mutable default lists, no
    methods that mutate state).
    """

    model_id: str = DEFAULT_MODEL_ID
    patch_size: int = DEFAULT_PATCH_SIZE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    # First-stage training freezes the backbone; Phase 1-2b unfreezes after N
    # epochs per the plan's "2-epoch freeze then low-LR fine-tune" policy.
    freeze: bool = True
    # "float32" | "float16" | "bfloat16". fp32 during backbone-only dev to
    # keep comparison with DETR head baselines clean; switch to bfloat16 in
    # the training config if memory-bound.
    dtype: str = "float32"
    # -1 = final hidden state. Integer ≥ 0 taps a specific transformer block
    # (Phase 1-2b may use multi-scale taps; for now we only expose the last).
    output_layer: int = -1

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {_SUPPORTED_DTYPES}, got {self.dtype!r}"
            )


class DinoV3Backbone:
    """Emit (B, C, H_patch, W_patch) features from a DINOv3 ViT."""

    def __init__(self, config: DinoV3BackboneConfig | None = None) -> None:
        self.config = config or DinoV3BackboneConfig()
        self._model: Any = None  # torch.nn.Module once loaded

    # -- Shape math: callable without torch --------------------------------

    def patch_grid_shape(self, image_height: int, image_width: int) -> tuple[int, int]:
        """Expected (H_patch, W_patch) for an image of the given pixel size."""
        p = self.config.patch_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError(
                f"image size must be positive, got ({image_height}, {image_width})"
            )
        if image_height % p != 0 or image_width % p != 0:
            raise ValueError(
                f"image size ({image_height}, {image_width}) not divisible by "
                f"patch_size={p}. DINOv3 ViT-B/16 requires patch-multiple inputs."
            )
        return (image_height // p, image_width // p)

    def num_patches(self, image_height: int, image_width: int) -> int:
        h, w = self.patch_grid_shape(image_height, image_width)
        return h * w

    # -- Actual model loading / forward: needs torch + transformers --------

    def load(self) -> Any:
        """Download or materialize the DINOv3 weights.

        Idempotent — second calls return the cached model. Raises
        `ImportError` if torch/transformers aren't installed, and surfaces
        the HF gated-access error verbatim if HF_TOKEN / license accept is
        missing (that's the signal the caller needs to fix their env).
        """
        if self._model is not None:
            return self._model

        try:
            import torch  # noqa: F401
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "DinoV3Backbone.load() requires torch and transformers. "
                "Install via `pip install -e .[dev]` or the training extras."
            ) from e

        kwargs: dict[str, Any] = {}
        if self.config.dtype != "float32":
            import torch

            kwargs["torch_dtype"] = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.config.dtype]

        model = AutoModel.from_pretrained(self.config.model_id, **kwargs)
        if self.config.freeze:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        self._model = model
        return model

    def forward(self, pixel_values: Any) -> Any:
        """Run backbone and reshape patch tokens into (B, C, Hp, Wp).

        pixel_values: torch.Tensor of shape (B, 3, H, W), dtype matching
        `config.dtype`. Caller is responsible for the ImageNet-style
        normalization DINOv3 expects (mean/std — see preprocessing/).
        """
        model = self.load()

        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                f"pixel_values must be (B, 3, H, W); got shape {tuple(pixel_values.shape)}"
            )
        B, _, H, W = pixel_values.shape
        Hp, Wp = self.patch_grid_shape(int(H), int(W))
        num_patch = Hp * Wp

        need_hidden = self.config.output_layer != -1
        output = model(
            pixel_values=pixel_values,
            output_hidden_states=need_hidden,
        )
        if need_hidden:
            # hidden_states[0] is embeddings; blocks are indexed from 1.
            hidden = output.hidden_states[self.config.output_layer]
        else:
            hidden = output.last_hidden_state

        # DINOv3 layout: [CLS, register_1..K, patch_1..Hp*Wp]. Slicing the
        # tail is robust to register-count changes across checkpoints.
        if hidden.shape[1] < num_patch:
            raise RuntimeError(
                f"backbone returned {hidden.shape[1]} tokens but expected at "
                f"least {num_patch} patches for input {H}x{W} (patch={self.config.patch_size})."
            )
        patch_tokens = hidden[:, -num_patch:, :]  # (B, Hp*Wp, C)
        features = (
            patch_tokens.transpose(1, 2).reshape(B, -1, Hp, Wp).contiguous()
        )
        return features

    __call__ = forward
