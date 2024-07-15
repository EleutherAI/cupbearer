import torch 
from cupbearer.detectors.statistical.statistical import StatisticalDetector
import numpy as np
from sklearn.ensemble import IsolationForest

class IsoForestDetector(StatisticalDetector):
    n_estimators = 20
    models = {}

    def init_variables(self, activation_sizes: dict[str, torch.Tensor], device):
        self.activations = {}
        for k, size in activation_sizes.items():
            self.activations = {
                k: torch.empty((0, size[-1]), device = 'cpu')
                for k, size in activation_sizes.items()
            }

    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            activation = activation.view(-1, activation.shape[-1]).cpu()
            if k in self.activations:
                self.activations[k] = torch.cat([self.activations[k], activation], dim = 0)

    def train(self, trusted_data, untrusted_data, **kwargs):

        super().train(trusted_data = trusted_data, untrusted_data = untrusted_data, **kwargs)

        for k, data in self.activations.items():
            data_np = data.numpy()
            model = IsolationForest(n_estimators = self.n_estimators)
            model.fit(data_np)
            self.models[k] = model

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        batch_size = next(iter(activations.values())).shape[0]

        distances = {}
        for k, activation in activations.items():

            activation = activation.view(-1, activation.shape[-1])

            data_np = activation.cpu().numpy()
            scores = self.models[k].decision_function(data_np)
            distances[k] = torch.tensor(-scores, dtype = torch.float32).view(batch_size, -1).mean(-1)

        return distances

    def _get_trained_variables(self, saving: bool = False):
        return {"Models": self.models}

    def _set_trained_variables(self, variables):
        self.models = variables["Models"]
