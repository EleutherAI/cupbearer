import torch
from typing import Any, Dict, Literal

from cupbearer.detectors.activation_based import ActivationBasedDetector
from sae.sae import EncoderOutput

class ScaledMeanDifferenceDetector(ActivationBasedDetector):
    def __init__(self, *args, metric: Literal['l0', 'l2'] = 'l0', **kwargs):
        super().__init__(*args, **kwargs)
        self.means = {}
        self.stds = {}
        self.metric = metric
        
    def _train(self, trusted_dataloader: torch.utils.data.DataLoader, untrusted_dataloader: torch.utils.data.DataLoader = None, **kwargs):
        self.means = {}
        self.stds = {}
        self.nonzero_indices = {}
        
        for batch, features in trusted_dataloader:
            for name, activation in features.items():
                values = activation

                if name not in self.means:
                    self.means[name] = torch.zeros_like(values[0])
                    self.stds[name] = torch.zeros_like(values[0])
                    if self.metric == 'l0':
                        self.nonzero_indices[name] = set()

                self.means[name] += values.sum(dim=0)
                self.stds[name] += (values ** 2).sum(dim=0)
                
                if self.metric == 'l0':
                    self.nonzero_indices[name].update(values.nonzero()[:, 1].tolist())

        num_samples = len(trusted_dataloader.dataset)
        for name in self.means:
            self.means[name] /= num_samples
            self.stds[name] = torch.sqrt(self.stds[name] / num_samples - self.means[name] ** 2)
            self.stds[name] = torch.clamp(self.stds[name], min=1e-8)  # Avoid division by zero
            
            if self.metric == 'l0':
                self.nonzero_indices[name] = torch.tensor(list(self.nonzero_indices[name]), dtype=torch.long)

    def _compute_layerwise_scores(self, inputs: Any, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores = {}
        for name, activation in features.items():
            if self.metric == 'l2':
                diff = activation - self.means[name]
                scaled_diff = diff / self.stds[name]
                scores[name] = torch.mean(scaled_diff ** 2, dim=-1)
            elif self.metric == 'l0':
                nonzero = activation.nonzero()
                batch_indices = nonzero[:, 0]
                feature_indices = nonzero[:, 1]
                mask = ~torch.isin(feature_indices, self.nonzero_indices[name].to(activation.device))
                counts = torch.bincount(batch_indices[mask], minlength=activation.shape[0])
                scores[name] = counts.float()

        return scores

    def _get_trained_variables(self):
        variables = {
            "means": self.means,
            "stds": self.stds,
        }
        if self.metric == 'l0':
            variables["nonzero_indices"] = self.nonzero_indices
        return variables

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.stds = variables["stds"]
        if self.metric == 'l0':
            self.nonzero_indices = variables["nonzero_indices"]