"""Densification mode contracts shared by training and benchmark entrypoints."""

from __future__ import annotations

from typing import Iterable

STANDARD_MODE = "standard_3dgs"
DENSIFICATION_MODE_CHOICES = (
    "standard",
    STANDARD_MODE,
    "patch_guided",
    "patch_guided_semantic",
    "patch_reallocate",
    "patch_reallocate_semantic",
)
PATCH_DENSIFICATION_MODES = frozenset(
    {
        "patch_guided",
        "patch_guided_semantic",
        "patch_reallocate",
        "patch_reallocate_semantic",
    }
)
SEMANTIC_DENSIFICATION_MODES = frozenset({"patch_guided_semantic", "patch_reallocate_semantic"})
REALLOCATE_DENSIFICATION_MODES = frozenset({"patch_reallocate", "patch_reallocate_semantic"})


def normalize_densification_mode(mode: str) -> str:
    """Return the canonical mode name accepted by downstream contracts."""
    if mode == "standard":
        return STANDARD_MODE
    if mode not in DENSIFICATION_MODE_CHOICES:
        raise ValueError(f"未知 densification mode: {mode}")
    return mode


def uses_patch_densifier(mode: str) -> bool:
    return normalize_densification_mode(mode) in PATCH_DENSIFICATION_MODES


def uses_semantic_importance(mode: str) -> bool:
    return normalize_densification_mode(mode) in SEMANTIC_DENSIFICATION_MODES


def uses_reallocation(mode: str) -> bool:
    return normalize_densification_mode(mode) in REALLOCATE_DENSIFICATION_MODES


def validate_semantic_importance_root(modes: Iterable[str], semantic_importance_root: str) -> None:
    missing = sorted({normalize_densification_mode(mode) for mode in modes if uses_semantic_importance(mode)})
    if missing and not semantic_importance_root:
        joined = ", ".join(missing)
        raise ValueError(f"{joined} 需要传入 --semantic-importance-root")
