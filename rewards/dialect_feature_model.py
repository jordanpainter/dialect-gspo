#!/usr/bin/env python
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiheadDialectFeatureModel(BertPreTrainedModel):
    """
    Multi-label dialect feature classifier.

    - Backbone: BERT
    - Representation: CLS token embedding
    - Output: num_features logits, one per dialect feature
    - Loss: BCEWithLogitsLoss for multi-label classification
    """

    def __init__(self, config, num_features: int = 135, pos_weight=None):
        super().__init__(config)

        self.num_features = num_features
        self.encoder = BertModel(config)

        hidden = config.hidden_size
        proj = hidden // 2

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden, proj),
            nn.LayerNorm(proj),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.classifier = nn.Linear(proj, num_features)

        self.temperature_scales = nn.Parameter(torch.ones(num_features))
        self.use_temperature_scaling = False

        self.register_buffer("pos_weight", pos_weight)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)

        shared = self.feature_extractor(cls)
        logits = self.classifier(shared)

        if self.use_temperature_scaling:
            logits = logits / self.temperature_scales.clamp(min=1e-6)

        loss = None
        if labels is not None:
            labels = labels.float()
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def enable_temperature_scaling(self):
        self.use_temperature_scaling = True

    def disable_temperature_scaling(self):
        self.use_temperature_scaling = False

    @torch.no_grad()
    def calibrate_temperature(self, dataloader, device="cpu", t_min=0.5, t_max=3.0, steps=50, n_bins=10):
        """
        Calibrate per-feature temperatures via simple grid search minimizing ECE.
        """
        self.eval()
        self.to(device)

        all_logits = []
        all_labels = []

        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device).float()

            out = self.encoder(input_ids=ids, attention_mask=att, return_dict=True)
            cls = out.last_hidden_state[:, 0]
            cls = self.dropout(cls)
            shared = self.feature_extractor(cls)
            logits = self.classifier(shared)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labs.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        temps = torch.linspace(t_min, t_max, steps)

        for i in range(self.num_features):
            best_temp = 1.0
            best_ece = float("inf")

            li = all_logits[:, i]
            yi = all_labels[:, i]

            for t in temps:
                probs = torch.sigmoid(li / t)
                ece = self._calculate_ece(probs, yi, n_bins=n_bins)
                if ece < best_ece:
                    best_ece = ece
                    best_temp = float(t.item())

            self.temperature_scales.data[i] = best_temp

        self.use_temperature_scaling = True
        avg_t = float(self.temperature_scales.mean().item())
        print(f"[Temperature calibrated] avg temp = {avg_t:.2f}")

    def _calculate_ece(self, probs, labels, n_bins=10):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (probs > low) & (probs <= high)
            frac = in_bin.float().mean()
            if frac.item() > 0:
                acc = labels[in_bin].float().mean()
                conf = probs[in_bin].mean()
                ece += torch.abs(acc - conf) * frac

        return float(ece.item())


@dataclass
class DialectFeatureTrainingArguments(TrainingArguments):
    calibrate_temperature: bool = False