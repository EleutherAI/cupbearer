import torch
import umap
import numpy as np

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


def _pinv(C, rcond, dtype=torch.float64):
    # Workaround for pinv not being supported on MPS
    if C.is_mps:
        return (
            torch.linalg.pinv(C.cpu().to(dtype), rcond=rcond, hermitian=True)
            .to(C.dtype)
            .to(C.device)
        )
    return torch.linalg.pinv(C.to(dtype), rcond=rcond, hermitian=True).to(C.dtype)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(
        self, rcond: float = 1e-3, relative: bool = False, **kwargs
    ):
        self.inv_covariances = {
            k: _pinv(C, rcond) for k, C in self.covariances["trusted"].items()
        }
        self.inv_diag_covariances = None
        if relative:
            self.inv_diag_covariances = {
                k: torch.where(torch.diag(C) > rcond, 1 / torch.diag(C), 0)
                for k, C in self.covariances["trusted"].items()
            }

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        inv_diag_covariance = None
        if self.inv_diag_covariances is not None:
            inv_diag_covariance = self.inv_diag_covariances[name]

        distance = mahalanobis(
            activation,
            self.means["trusted"][name],
            self.inv_covariances[name],
            inv_diag_covariance=inv_diag_covariance,
        )

        # Normalize by the number of dimensions (no sqrt since we're using *squared*
        # Mahalanobis distance)
        return distance / self.means["trusted"][name].shape[0]

    def _get_trained_variables(self):
        return {
            "means": self.means,
            "inv_covariances": self.inv_covariances,
            "inv_diag_covariances": self.inv_diag_covariances,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.inv_covariances = variables["inv_covariances"]
        self.inv_diag_covariances = variables["inv_diag_covariances"]

class UMAPMahalanobisDetector(MahalanobisDetector):
    """
    Mahalanobis detector that learns a UMAP embedding on train data and transforms test data using this embedding.
    """
    def __init__(self, n_components=5, n_neighbors=15, min_dist=0.1, **kwargs):
        super().__init__(**kwargs)
        self.umap_reducers = {}
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def post_covariance_training(self, rcond: float = 1e-3, relative: bool = False, **kwargs):
        super().post_covariance_training(rcond, relative, **kwargs)
        
        for name, activations in self.activations["trusted"].items():
            reducer = umap.UMAP(n_components=self.n_components, 
                                n_neighbors=self.n_neighbors, 
                                min_dist=self.min_dist)
            reduced_activations = reducer.fit_transform(activations.cpu().numpy())
            self.umap_reducers[name] = reducer
            
            # Update means and covariances with reduced activations
            self.means["trusted"][name] = torch.tensor(reduced_activations.mean(axis=0), device=self.device)
            cov = torch.tensor(np.cov(reduced_activations.T), device=self.device)
            self.covariances["trusted"][name] = cov
            self.inv_covariances[name] = _pinv(cov, rcond)

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        # Transform the activation using the learned UMAP embedding
        reduced_activation = torch.tensor(
            self.umap_reducers[name].transform(activation.cpu().numpy()),
            device=self.device
        )
        
        return super()._individual_layerwise_score(name, reduced_activation)

    def _get_trained_variables(self):
        variables = super()._get_trained_variables()
        variables["umap_reducers"] = self.umap_reducers
        return variables

    def _set_trained_variables(self, variables):
        super()._set_trained_variables(variables)
        self.umap_reducers = variables["umap_reducers"]