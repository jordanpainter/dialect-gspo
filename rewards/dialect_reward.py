from __future__ import annotations

import os
from typing import List, Optional

import torch

from .dialect_reward_model import DialectDensityScorer

_SCORER: Optional[DialectDensityScorer] = None


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _default_device() -> str:
    if torch.cuda.is_available():
        lr = _get_local_rank()
        torch.cuda.set_device(lr)
        return f"cuda:{lr}"
    return "cpu"


def _get_scorer() -> DialectDensityScorer:
    global _SCORER
    if _SCORER is None:
        model_path = os.environ.get("DIALECT_REWARD_MODEL", "srirag/feature-identifier")
        device = os.environ.get("DIALECT_REWARD_DEVICE")
        if device is None:
            device = _default_device()

        max_length = int(os.environ.get("DIALECT_REWARD_MAX_LENGTH", "256"))
        _SCORER = DialectDensityScorer(
            model_path=model_path,
            device=device,
            max_length=max_length,
        )
    return _SCORER


def dialect_density(texts: List[str]) -> List[float]:
    """
    Returns dialect density in [0, 1] for each text.
    """
    scorer = _get_scorer()
    densities = scorer.score_density(texts)

    if isinstance(densities, torch.Tensor):
        densities = densities.detach().cpu().tolist()

    return [float(x) for x in densities]


def dialect_raw_score(texts: List[str]) -> List[float]:
    """
    Returns raw expected feature count for each text.
    """
    scorer = _get_scorer()
    raw = scorer.score_raw(texts)

    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()

    return [float(x) for x in raw]


def dialect_density_gain(generated_texts: List[str], base_texts: List[str]) -> List[float]:
    """
    Returns density(generation) - density(base) for each paired example.
    """
    if len(generated_texts) != len(base_texts):
        raise ValueError(
            f"generated_texts and base_texts must have the same length, got "
            f"{len(generated_texts)} and {len(base_texts)}"
        )

    scorer = _get_scorer()
    details = scorer.compare_density(generated_texts, base_texts)
    gain = details["gain"]

    if isinstance(gain, torch.Tensor):
        gain = gain.detach().cpu().tolist()

    return [float(x) for x in gain]