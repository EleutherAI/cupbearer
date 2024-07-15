import torch

from cupbearer.detectors.statistical.helpers import local_outlier_factor
from cupbearer.detectors.statistical.statistical import StatisticalDetector, ActivationCovarianceBasedDetector

class LOFDetector(ActivationCovarianceBasedDetector):
    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        self._activations = {
            k: torch.empty((0, torch.tensor(size).prod()), device=device)
            for k, size in activation_sizes.items()
        }
    
    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            self._activations[k] = torch.cat([self._activations[k], activation.view(activation.shape[0], -1)], dim=0)

    def train(self, trusted_data, untrusted_data, **kwargs):
        StatisticalDetector.train(
            self, trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process
        self.activations = self._activations

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        batch_size = next(iter(activations.values())).shape[0]
        activations = {
            k: v.view(batch_size, -1)
            for k, v in activations.items()
        }
        scores = {
            k: self._individual_layerwise_score(k, v) for k, v in activations.items()
        }
        return scores


    def _individual_layerwise_score(self, name: str, activations: torch.Tensor):
        reshaped_activations = activations.view(activations.shape[0], -1)
        score = local_outlier_factor(reshaped_activations, self.activations[name])
        return score

    def _get_trained_variables(self, saving: bool = False):
        return {
            "activations": self.activations,
        }

    def _set_trained_variables(self, variables):
        self.activations = variables["activations"]

    def post_covariance_training(self, **kwargs):
        pass