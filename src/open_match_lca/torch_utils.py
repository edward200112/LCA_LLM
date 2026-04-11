from __future__ import annotations

import os

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def resolve_torch_device(preferred: str | None = None) -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch is None:  # pragma: no cover
        return "cpu"

    preferred_device = (preferred or "").strip().lower()
    if preferred_device and preferred_device != "auto":
        if preferred_device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError(
                "Requested torch device 'mps', but Metal Performance Shaders is not available."
            )
        if preferred_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("Requested torch device 'cuda', but CUDA is not available.")
        if preferred_device == "cpu":
            return "cpu"
        raise RuntimeError(f"Unsupported torch device setting: {preferred}")

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
