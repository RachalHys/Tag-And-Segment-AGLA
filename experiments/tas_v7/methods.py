"""Pure next-token fusion for the frozen-prefix TAS v7 experiment.

Scores are returned BEFORE temperature, support filtering, processors or
sampling.  Each input must describe the SAME prompt and generated prefix.
Probability bounds apply here, not to the postprocessed sampling distribution.
No tensor is modified in place.  Importing this module does not import torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class MethodSpec:
    id: str
    views: tuple[str, ...]
    fusion: str
    support: bool
    alpha: float = 2.0
    beta: float = 0.5
    gamma: float = 0.0
    eta: float = 0.4
    omega: float = 0.75
    delta: float = 0.2
    kappa: float = 1.0
    cap: float = 1.5
    protocol: str = "HARMONIZED"
    context: str | None = None
    control: str | None = None

    def __post_init__(self):
        if not self.id or not self.views or self.views[0] != "ORIG":
            raise ValueError("A method needs an id and ORIG as its first view")
        if len(self.views) != len(set(self.views)):
            raise ValueError(
                "View names must be unique; use context='ORIG' for duplicate control"
            )
        if self.fusion not in {
            "native",
            "scale",
            "tag",
            "tag_normalized",
            "linear",
            "mixture_object",
            "mixture",
            "selective",
            "linear_context",
            "mixture_context",
            "selective_context",
        }:
            raise ValueError(f"Unknown fusion {self.fusion!r}")
        if self.protocol not in {"LEGACY", "HARMONIZED"}:
            raise ValueError("protocol must be LEGACY or HARMONIZED")
        for name in ("alpha", "beta", "gamma", "eta", "omega", "delta", "kappa", "cap"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if (
            not 0 <= self.beta <= 1
            or not 0 <= self.eta <= 1
            or not 0 <= self.omega <= 1
        ):
            raise ValueError("beta, eta and omega must be in [0,1]")
        if self.kappa < 0 or self.cap < 0 or self.delta < 0:
            raise ValueError("kappa, cap and delta must be nonnegative")
        if (
            self.fusion
            not in {
                "native",
                "scale",
                "linear_context",
                "mixture_context",
                "selective_context",
            }
            and "OBJ-H" not in self.views
        ):
            raise ValueError(f"{self.fusion} requires OBJ-H")
        if self.fusion in {
            "linear",
            "mixture",
            "selective",
            "linear_context",
            "mixture_context",
            "selective_context",
        }:
            context = self.context_view
            if context not in self.views:
                raise ValueError("Context must name a supplied view")

    @property
    def context_view(self) -> str | None:
        return (
            self.context
            if self.context is not None
            else (self.views[2] if len(self.views) > 2 else None)
        )

    def to_dict(self) -> dict:
        return asdict(self)


def initial_methods() -> list[MethodSpec]:
    """The exact ordered fourteen configurations in tas_idea.md v7."""
    orig, obj, bh, fg = (
        ("ORIG",),
        ("ORIG", "OBJ-H"),
        ("ORIG", "OBJ-H", "BH"),
        ("ORIG", "OBJ-H", "FG-U"),
    )
    return [
        MethodSpec("E0", orig, "native", False, protocol="LEGACY"),
        MethodSpec("C0", orig, "native", True),
        MethodSpec("C1", orig, "scale", True),
        MethodSpec("C2", obj, "tag_normalized", True),
        MethodSpec("E1", obj, "tag", True, protocol="LEGACY"),
        MethodSpec("E2", bh, "linear", True, gamma=0.5),
        MethodSpec("E3", bh, "linear", True, gamma=-0.5),
        MethodSpec("E4", fg, "linear", True, gamma=0.5),
        MethodSpec("E5", fg, "linear", True, gamma=-0.5),
        MethodSpec("F0", obj, "mixture_object", True),
        MethodSpec("F1", bh, "mixture", True),
        MethodSpec("F2", fg, "mixture", True),
        MethodSpec("F3", bh, "selective", True),
        MethodSpec("F4", fg, "selective", True),
    ]


def method_by_id(method_id: str) -> MethodSpec:
    for method in initial_methods():
        if method.id == method_id:
            return method
    raise KeyError(f"Unknown initial method {method_id!r}")


def harmonized_baselines() -> list[MethodSpec]:
    """Optional same-protocol controls; never replace the native E0/E1 records."""
    return [
        replace(method_by_id("E0"), id="E0-H", protocol="HARMONIZED"),
        replace(method_by_id("E1"), id="E1-H", protocol="HARMONIZED"),
    ]


def _weighted_logsumexp(logs, weights):
    import torch

    active = [
        (value + math.log(weight)) for value, weight in zip(logs, weights) if weight > 0
    ]
    if not active:
        raise ValueError("Mixture weights sum to zero")
    return (
        active[0]
        if len(active) == 1
        else torch.logsumexp(torch.stack(active, dim=0), dim=0)
    )


def fuse_logits(
    logits_by_view: Mapping[str, "torch.Tensor"], method: MethodSpec
) -> "torch.Tensor":
    """Fuse raw [batch,vocabulary] logits without applying a support mask.

    Native TAG arithmetic deliberately remains in the backbone dtype, including
    multiplication then addition.  New linear and probability math use FP32.
    """
    import torch
    import torch.nn.functional as F

    missing = set(method.views) - set(logits_by_view)
    if missing:
        raise KeyError(f"Missing views for {method.id}: {sorted(missing)}")
    z0 = logits_by_view["ORIG"]
    if not isinstance(z0, torch.Tensor) or z0.ndim != 2 or not z0.is_floating_point():
        raise ValueError("Logits must be floating [batch,vocabulary] tensors")
    for view in method.views:
        other = logits_by_view[view]
        if (
            other.shape != z0.shape
            or other.device != z0.device
            or other.dtype != z0.dtype
        ):
            raise ValueError("All view logits must share shape, device and dtype")
    if method.fusion == "native":
        return z0
    if method.fusion == "scale":
        return (1.0 + method.alpha) * z0.float()
    if method.fusion == "linear_context":
        context = logits_by_view[method.context_view]
        # Replace object evidence with ORIG: preserve context strength and the
        # full family's coefficient sum instead of retuning either ablation.
        return (1 + method.alpha) * z0.float() + method.gamma * (
            context.float() - z0.float()
        )
    zo = (
        z0
        if method.fusion in {"mixture_context", "selective_context"}
        else logits_by_view["OBJ-H"]
    )
    if method.fusion == "tag":
        if method.protocol == "HARMONIZED":
            return z0.float() + method.alpha * zo.float()
        # Matches eval_tas/sample.py: FP32 scalar -> native dtype, then mul/add.
        alpha = torch.tensor(method.alpha, dtype=torch.float32, device=z0.device).to(
            z0.dtype
        )
        return z0 + alpha * zo
    if method.fusion == "tag_normalized":
        if method.alpha == -1:
            raise ValueError("Normalized TAG requires alpha != -1")
        return (z0.float() + method.alpha * zo.float()) / (1.0 + method.alpha)
    context = logits_by_view.get(method.context_view)
    if method.fusion == "linear":
        return (
            z0.float()
            + method.alpha * zo.float()
            + method.gamma * (context.float() - z0.float())
        )

    lp0, lpo = F.log_softmax(z0.float(), dim=-1), F.log_softmax(zo.float(), dim=-1)
    if method.fusion == "mixture_object":
        return _weighted_logsumexp((lp0, lpo), (1 - method.eta, method.eta))
    lpc = F.log_softmax(context.float(), dim=-1)
    lq = _weighted_logsumexp((lpo, lpc), (method.omega, 1 - method.omega))
    if (
        method.fusion in {"selective", "selective_context"}
        and method.kappa > 0
        and method.cap > 0
    ):
        dominance = torch.clamp(
            lpc - torch.maximum(lp0, lpo) - method.delta, min=0, max=method.cap
        )
        penalized = lq - method.kappa * dominance
        lq = penalized - torch.logsumexp(penalized, dim=-1, keepdim=True)
    return _weighted_logsumexp((lp0, lq), (1 - method.eta, method.eta))


def plausibility_mask(
    original_logits: "torch.Tensor", beta: float, mode: str = "legacy"
) -> "torch.Tensor":
    """Original-view support, independent of how many branches were executed.

    Legacy reproduces the FP32 log(beta) -> native cast before adding the
    native maximum.  Harmonized compares in FP32.  A zero beta disables the
    additional constraint; forbidden tokens are still managed by the caller.
    """
    import torch

    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0,1]")
    if original_logits.ndim != 2 or not original_logits.is_floating_point():
        raise ValueError("original_logits must be floating [batch,vocabulary]")
    if mode.lower() not in {"legacy", "harmonized"}:
        raise ValueError("mask mode must be legacy or harmonized")
    if beta == 0:
        return torch.ones_like(original_logits, dtype=torch.bool)
    log_beta = torch.log(
        torch.tensor(beta, device=original_logits.device, dtype=torch.float32)
    )
    values = original_logits if mode.lower() == "legacy" else original_logits.float()
    cutoff = log_beta.to(values.dtype) + values.max(dim=-1, keepdim=True).values
    return values >= cutoff


def build_controls(method: MethodSpec) -> list[MethodSpec]:
    """Filter-off and duplicate-context method controls for a chosen finalist.

    Black/random controls alter the supplied pixels and are constructed in
    views.py; they MUST receive distinct run configuration identities.
    """
    controls = [
        replace(method, id=f"{method.id}_NOFILTER", support=False, control="filter_off")
    ]
    if method.context_view is not None:
        views = tuple(
            view for view in method.views if view == "ORIG" or view == "OBJ-H"
        )
        controls.append(
            replace(
                method,
                id=f"{method.id}_DUP_ORIG",
                views=views,
                context="ORIG",
                control="duplicate_context",
            )
        )
    return controls
