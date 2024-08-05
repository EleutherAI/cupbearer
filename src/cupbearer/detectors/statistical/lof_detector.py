import torch

from cupbearer.detectors.statistical.helpers import local_outlier_factor
from cupbearer.detectors.statistical.statistical import ActivationCovarianceBasedDetector, StatisticalDetector
import umap

class LOFDetector(ActivationCovarianceBasedDetector):
    use_trusted = True
    use_untrusted = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activations = {}

    def init_variables(self, activation_sizes: dict[str, torch.Size], device, case: str):
        self._activations[case] = {
            k: torch.empty((0, torch.tensor(size).prod()), device=device)
            for k, size in activation_sizes.items()
        }
    
    def batch_update(self, activations: dict[str, torch.Tensor], case: str):
        for k, activation in activations.items():
            self._activations[case][k] = torch.cat([self._activations[case][k], activation.view(activation.shape[0], -1)], dim=0)

    def train(self, trusted_data, untrusted_data, **kwargs):
        StatisticalDetector.train(
            self, trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process
        self.activations = self._activations

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        score = local_outlier_factor(activation, self.activations['trusted'][name])
        return score

    def _get_trained_variables(self, saving: bool = False):
        return {
            "activations": self.activations,
        }

    def _set_trained_variables(self, variables):
        self.activations = variables["activations"]

    def post_covariance_training(self, **kwargs):
        pass

class UMAPLOFDetector(LOFDetector):
    def __init__(self, n_components=5, n_neighbors=15, min_dist=0.1, **kwargs):
        super().__init__(**kwargs)
        self.umap_reducers = {}
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def post_covariance_training(self, **kwargs):
        super().post_covariance_training(**kwargs)
        
        for name, activations in self.activations["trusted"].items():
            reducer = umap.UMAP(n_components=self.n_components, 
                                n_neighbors=self.n_neighbors, 
                                min_dist=self.min_dist)
            reduced_activations = reducer.fit_transform(activations.cpu().numpy())
            self.umap_reducers[name] = reducer
            
            # Update LOF with reduced activations
            self.lof[name].fit(reduced_activations)

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        reduced_activation = self.umap_reducers[name].transform(activation.cpu().numpy())
        return -self.lof[name].score_samples(reduced_activation)

    def _get_trained_variables(self):
        variables = super()._get_trained_variables()
        variables["umap_reducers"] = self.umap_reducers
        return variables

    def _set_trained_variables(self, variables):
        super()._set_trained_variables(variables)
        self.umap_reducers = variables["umap_reducers"]