import torch 
from cupbearer.detectors.statistical.statistical import StatisticalDetector
import numpy as np
from sklearn.ensemble import IsolationForest

class IsoForestDetector(StatisticalDetector):
    use_trusted = True
    use_untrusted = False
    n_estimators = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = {}
        self._activations = {}

    def init_variables(self, activation_sizes: dict[str, torch.Size], device, case: str):
        self._activations[case] = {
            k: torch.empty((0, size[-1]), device='cpu')
            for k, size in activation_sizes.items()
        }

    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        for k, activation in activations.items():
            activation = activation.view(-1, activation.shape[-1]).cpu()
            self._activations[case][k] = torch.cat([self._activations[case][k], activation], dim=0)

    def _train(self, trusted_dataloader, untrusted_dataloader, **kwargs):

        super()._train(trusted_dataloader=trusted_dataloader, untrusted_dataloader=untrusted_dataloader, **kwargs)

        for case, data_dict in self._activations.items():
            if case not in self.models:
                self.models[case] = {}
            for k, data in data_dict.items():
                data_np = data.to(torch.float32).numpy()
                model = IsolationForest(n_estimators=self.n_estimators)
                model.fit(data_np)
                self.models[case][k] = model

    def _compute_layerwise_scores(self, inputs, features):
        batch_size = next(iter(features.values())).shape[0]

        distances = {}
        for k, feature in features.items():

            feature = feature.view(-1, feature.shape[-1])

            data_np = feature.cpu().numpy()
            scores = self.models['trusted'][k].decision_function(data_np)
            distances[k] = torch.tensor(-scores, dtype = torch.float32).view(batch_size, -1).mean(-1)

        return distances

    def _get_trained_variables(self, saving: bool = False):
        return {"models": self.models}

    def _set_trained_variables(self, variables):
        self.models = variables["models"]