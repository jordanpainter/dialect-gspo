from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from .dialect_feature_model import MultiheadDialectFeatureModel


class DialectDensityScorer:
    """
    Loads the trained dialect feature classifier and exposes reusable scoring methods.

    For each text, we compute:
    - logits for each dialect feature
    - probabilities via sigmoid
    - raw_score = sum(feature probabilities)
    - density = raw_score / num_features

    density is in [0, 1] and is the cleanest quantity to compare between:
    - generation
    - base output
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultiheadDialectFeatureModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.num_features = int(self.model.num_features)

    def to(self, device: str) -> "DialectDensityScorer":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

    @torch.no_grad()
    def predict_logits(self, texts: List[str]) -> torch.Tensor:
        inputs = self._tokenize(texts)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs.logits.detach()

    @torch.no_grad()
    def predict_feature_probabilities(self, texts: List[str]) -> torch.Tensor:
        logits = self.predict_logits(texts)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def score_raw(self, texts: List[str]) -> torch.Tensor:
        """
        Returns the expected number of active dialect features per text.
        Shape: (batch,)
        """
        probs = self.predict_feature_probabilities(texts)
        return probs.sum(dim=1)

    @torch.no_grad()
    def score_density(self, texts: List[str]) -> torch.Tensor:
        """
        Returns dialect density per text in [0, 1].
        Shape: (batch,)
        """
        raw = self.score_raw(texts)
        return raw / float(self.num_features)

    @torch.no_grad()
    def score_details(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Convenience method for debugging or analysis.
        Returns:
        - feature_probs: (batch, num_features)
        - raw: (batch,)
        - density: (batch,)
        """
        probs = self.predict_feature_probabilities(texts)
        raw = probs.sum(dim=1)
        density = raw / float(self.num_features)
        return {
            "feature_probs": probs.detach().cpu(),
            "raw": raw.detach().cpu(),
            "density": density.detach().cpu(),
        }

    @torch.no_grad()
    def compare_density(self, generated_texts: List[str], base_texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Returns:
        - gen_density
        - base_density
        - gain = gen_density - base_density
        """
        gen_density = self.score_density(generated_texts)
        base_density = self.score_density(base_texts)
        gain = gen_density - base_density

        return {
            "gen_density": gen_density.detach().cpu(),
            "base_density": base_density.detach().cpu(),
            "gain": gain.detach().cpu(),
        }