import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.statistical.helpers import quantum_entropy
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class QuantumEntropyDetector(ActivationCovarianceBasedDetector):
    def __init__(self, activation_names: list[str], alpha: float, **kwargs):
        super().__init__(activation_names, **kwargs)
        self.alpha = alpha

    def post_covariance_training(self, rcond: float = 1e-5, **kwargs):
        whitening_matrices = {}
        for k, cov in self.covariances.items():
            # Compute decomposition
            eigs = torch.linalg.eigh(cov)

            # Zero entries corresponding to eigenvalues smaller than rcond
            vals_rsqrt = eigs.eigenvalues.rsqrt()
            vals_rsqrt[eigs.eigenvalues < rcond * eigs.eigenvalues.max()] = 0

            # PCA whitening
            # following https://doi.org/10.1080/00031305.2016.1277159
            # and https://stats.stackexchange.com/a/594218/319192
            # but transposed (sphering with x@W instead of W@x)
            whitening_matrices[k] = eigs.eigenvectors * vals_rsqrt.unsqueeze(0)
            assert torch.allclose(
                whitening_matrices[k], eigs.eigenvectors @ vals_rsqrt.diag()
            )
        self.whitening_matrices = whitening_matrices

    def batch_update_untrusted(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape
            self._activations[k] = torch.cat([self._activations[k], activation], dim=0)

    def train(self, trusted_data, untrusted_data, **kwargs):
        super().train(
            trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process
        with torch.inference_mode():   
            self.use_trusted = False

            self._activations = {k: torch.empty(0, self.means[k].shape[0], device=self.means[k].device) for k in self.means.keys()}

            data_loader = DataLoader(untrusted_data, batch_size=kwargs.get('batch_size', 1024), shuffle=False)
            for batch in tqdm(data_loader):
                activations = self.get_activations(batch)
                self.batch_update_untrusted(activations)

            whitened_activations = {
                k: torch.einsum(
                    "bi,ij->bj",
                    self._activations[k].flatten(start_dim=1) - self.means[k],
                    self.whitening_matrices[k],
                )
                for k in self._activations.keys()
            }
            whitened_activations = {
                k: whitened_activations[k].flatten(start_dim=1) - 
                whitened_activations[k].flatten(start_dim=1).mean(dim=0, keepdim=True) 
                for k in whitened_activations.keys()
            }

            self.untrusted_covariances = {k: whitened_activations[k].mT @ whitened_activations[k] for k in whitened_activations.keys()}

    def _individual_layerwise_score(self, name, activation):
        whitened_activations = torch.einsum(
            "bi,ij->bj",
            activation.flatten(start_dim=1) - self.means[name],
            self.whitening_matrices[name],
        )
        # TODO should possibly pass rank
        return quantum_entropy(whitened_activations, 
                               alpha=self.alpha,
                               batch_cov=self.untrusted_covariances[name])

    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "whitening_matrices": self.whitening_matrices,
            "untrusted_covariances": self.untrusted_covariances,
            "covariances": self.covariances,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.covariances = variables["covariances"]
        self.whitening_matrices = variables["whitening_matrices"]
        self.untrusted_covariances = variables["untrusted_covariances"]
