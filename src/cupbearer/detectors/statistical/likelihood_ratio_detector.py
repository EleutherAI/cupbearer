import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
import pdb

from cupbearer.utils.optimal_shrinkage import optimal_linear_shrinkage


from .statistical import ActivationCovarianceBasedDetector

def get_nonzero_basis(cov_matrix):
    """Gets the basis vectors of the covariance matrix with nonzero eigenvalues"""
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    mask = eigenvalues > 1.e-5
    filtered_eigenvectors = eigenvectors[:, mask]

    return filtered_eigenvectors

def project_data(data, basis):
    return torch.matmul(data, basis)

def project_cov_matrix(cov_matrix, basis):
    # basis should be a matrix where each column is a basis vector
    projected_cov_matrix = basis.T @ cov_matrix @ basis
    return (projected_cov_matrix + projected_cov_matrix.T) / 2

class LikelihoodRatioDetector(ActivationCovarianceBasedDetector):
    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        super().init_variables(activation_sizes, device)
        self._activations = {
            k: torch.empty(0, size[-1], device=device) 
            for k, size in activation_sizes.items()
        }
        self._activations_untrusted = {
            k: torch.empty(0, size[-1], device=device) 
            for k, size in activation_sizes.items()
        }

        # We don't actually use the online covariances, but we need to initialize them
        self._Cs = {
            k: v+1
            for k, v in self._Cs.items()
        }

    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape
            self._activations[k] = torch.cat([self._activations[k], activation], dim=0)

    def batch_update_untrusted(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            if activation.ndim == 3:
                activation = rearrange(
                    activation, "batch independent dim -> (batch independent) dim"
                )
            assert activation.ndim == 2, activation.shape
            self._activations_untrusted[k] = torch.cat([self._activations_untrusted[k], activation], dim=0)

    def post_covariance_training(self, **kwargs):
        pass

    def _train(self, trusted_data, untrusted_data, **kwargs):

        super()._train(trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs)

        means = {
            k: torch.mean(self._activations[k], dim=0) 
            for k in self._activations.keys()
        }
        covariances = {
            k: torch.cov(self._activations[k].T)
            for k in self._activations.keys()
        }

        self.trusted_basis = {
            k: get_nonzero_basis(covariances[k])
            for k in covariances.keys()
        }

        # Second pass for untrusted data
        with torch.inference_mode():
            data_loader = DataLoader(untrusted_data, batch_size=kwargs.get('batch_size', 1024), shuffle=False)
            for batch in tqdm(data_loader):
                activations = self.get_activations(batch)
                self.batch_update_untrusted(activations)

            means_untrusted = {
                k: torch.mean(self._activations_untrusted[k], dim=0) 
                for k in self._activations_untrusted.keys()
            }
            covariances_untrusted = {
                k: torch.cov(self._activations_untrusted[k].T)
                for k in self._activations_untrusted.keys()
            }

            self.untrusted_basis = {
                k: get_nonzero_basis(covariances_untrusted[k])
                for k in covariances_untrusted.keys()
            }

            self.preferred_basis = {
                k: self.trusted_basis[k] 
                if self.trusted_basis[k].shape[1] <= self.untrusted_basis[k].shape[1] 
                else self.untrusted_basis[k]
                for k in self.trusted_basis.keys()
            }

            self.means = {
                k: project_data(means[k], self.preferred_basis[k])
                for k in means.keys()
            }

            self.covariances = {
                k: optimal_linear_shrinkage(project_cov_matrix(covariances[k], self.preferred_basis[k]), len(self._activations[k]))
                for k in covariances.keys()
            }

            self.means_untrusted = {
                k: project_data(means_untrusted[k], self.preferred_basis[k])
                for k in means_untrusted.keys()
            }
            self.covariances_untrusted = {
                k: optimal_linear_shrinkage(project_cov_matrix(covariances_untrusted[k], self.preferred_basis[k]), len(self._activations_untrusted[k]))
                for k in covariances_untrusted.keys()
            }

            if any(torch.count_nonzero(C) == 0 for C in self.covariances_untrusted.values()):
                raise RuntimeError("All zero covariance matrix detected in untrusted data.")

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        if activation.ndim == 3:
            activation = rearrange(
                activation, "batch independent dim -> (batch independent) dim"
            )
        assert activation.ndim == 2, activation.shape

        trusted_dist = MultivariateNormal(self.means[name], self.covariances[name])
        untrusted_dist = MultivariateNormal(self.means_untrusted[name], self.covariances_untrusted[name])

        log_prob_trusted = trusted_dist.log_prob(
            project_data(activation, self.preferred_basis[name])
        )
        log_prob_untrusted = untrusted_dist.log_prob(
            project_data(activation, self.preferred_basis[name])
        )
        # log_prob_trusted = trusted_dist.log_prob(activation)
        # log_prob_untrusted = untrusted_dist.log_prob(activation)

        return log_prob_untrusted - log_prob_trusted
    
    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "covariances": self.covariances,
            "means_untrusted": self.means_untrusted,
            "covariances_untrusted": self.covariances_untrusted,
            "trusted_basis": self.trusted_basis,
            "untrusted_basis": self.untrusted_basis,
            "preferred_basis": self.preferred_basis,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.covariances = variables["covariances"]
        self.means_untrusted = variables["means_untrusted"]
        self.covariances_untrusted = variables["covariances_untrusted"]
        self.trusted_basis = variables["trusted_basis"]
        self.untrusted_basis = variables["untrusted_basis"]
        self.preferred_basis = variables["preferred_basis"]


class ExpectationMaximisationDetector(LikelihoodRatioDetector):
    def train(self, trusted_data, untrusted_data, iterations=100, **kwargs):
        super().train(trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs)
        self.activations = self._activations
        self.activations_untrusted = self._activations_untrusted

        for _ in range(iterations):
            # E-step: calculate responsibilities
            responsibilities = {}

            activations_trusted = {k: project_data(self.activations[k], self.preferred_basis[k]) for k in self.activations.keys()}
            activations_untrusted = {k: project_data(self.activations_untrusted[k], self.preferred_basis[k]) for k in self.activations_untrusted.keys()}

            for k in activations_untrusted.keys():
                trusted_dist = MultivariateNormal(self.means[k], self.covariances[k])
                untrusted_dist = MultivariateNormal(self.means_untrusted[k], self.covariances_untrusted[k])

                log_prob_trusted = trusted_dist.log_prob(activations_untrusted[k])
                log_prob_untrusted = untrusted_dist.log_prob(activations_untrusted[k])

                denominator = torch.logsumexp(torch.cat([log_prob_trusted.unsqueeze(1), log_prob_untrusted.unsqueeze(1)], dim=1), dim=1)

                responsibilities[k] = torch.exp(log_prob_trusted - denominator)

            avg_responsibilities = torch.cat([responsibilities[k].unsqueeze(1) for k in responsibilities.keys()], dim=1).mean(dim=1, keepdim=True)
            # M-step: update parameters
            for k in activations_untrusted.keys():
                self.means_untrusted[k] = ((1 - avg_responsibilities) * activations_untrusted[k]).sum(dim=0) / (1 - avg_responsibilities).sum()

                centered_untrusted = activations_untrusted[k] - self.means_untrusted[k]

                self.covariances_untrusted[k] = optimal_linear_shrinkage((centered_untrusted.T @ (centered_untrusted * (1 - avg_responsibilities))) / (1 - avg_responsibilities).sum(), (1 - avg_responsibilities).sum())


    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "covariances": self.covariances,
            "means_untrusted": self.means_untrusted,
            "covariances_untrusted": self.covariances_untrusted,
            "activations": self.activations,
            "activations_untrusted": self.activations_untrusted,
            "preferred_basis": self.preferred_basis,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.covariances = variables["covariances"]
        self.means_untrusted = variables["means_untrusted"]
        self.covariances_untrusted = variables["covariances_untrusted"]
        self.preferred_basis = variables["preferred_basis"]
        self.activations = variables["activations"]
        self.activations_untrusted = variables["activations_untrusted"]