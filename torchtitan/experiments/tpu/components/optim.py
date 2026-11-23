"""TPU optimizer config helpers."""

from torchtitan.components.optim import AdamW


def default_adamw(
    *,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    implementation: str = "fused",
) -> AdamW.Config:
  """Compatibility helper replacing removed ``optimizer.default_adamw``."""
  fused = implementation == "fused"
  return AdamW.Config(
      pattern=".*",
      lr=lr,
      betas=betas,
      eps=eps,
      weight_decay=weight_decay,
      fused=fused,
  )


__all__ = ["default_adamw"]
