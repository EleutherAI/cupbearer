from abc import ABC, abstractmethod

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Any, Tuple, Dict

import pdb

from tqdm import tqdm
from einops import rearrange, reduce
from sklearn.ensemble import IsolationForest
import plotly.express as px
import torch
from cupbearer import detectors, tasks, utils, scripts
from cupbearer.detectors.statistical.statistical import ActivationCovarianceBasedDetector
from cupbearer.data import HuggingfaceDataset, MixedData
from torch import Tensor, nn
from torch.utils.data import DataLoader
from datasets import concatenate_datasets


@contextmanager
def atp(model: nn.Module, noise_acts: Dict[str, Tensor] | Dict[str, Tuple[Tensor, Tensor]], *, head_dim: int = 0):
    """Perform attribution patching on `model` with `noise_acts`.

    This function adds forward and backward hooks to submodules of `model`
    so that when you run a forward pass, the relevant activations are cached
    in a dictionary, and when you run a backward pass, the gradients w.r.t.
    activations are used to compute approximate causal effects.

    Args:
        model (nn.Module): The model to patch.
        noise_acts (dict[str, Tensor]): A dictionary mapping (suffixes of) module
            paths to noise activations.
        head_dim (int): The size of each attention head, if applicable. When nonzero,
            the effects are returned with a head dimension.

    Example:
    ```python
    noise = {
        "model.encoder.layer.0.attention.self": torch.zeros(1, 12, 64, 64),
    }
    with atp(model, noise) as effects:
        probs = model(input_ids).logits.softmax(-1)
        probs[0].backward()

    # Use the effects
    ```
    """
    # Keep track of all the hooks we're adding to the model
    handles: list[nn.modules.module.RemovableHandle] = []

    # Keep track of activations from the forward pass
    mod_to_clean: dict[nn.Module, Tensor] = {}
    mod_to_noise: dict[nn.Module, Tensor] | dict[nn.Module, Tuple[Tensor, Tensor]] = {}

    # Dictionary of effects
    effects: dict[str, Tensor] = {}
    mod_to_name: dict[nn.Module, str] = {}

    # Backward hook
    def bwd_hook(module: nn.Module, grad_input: tuple[Tensor, ...] | Tensor, grad_output: tuple[Tensor, ...] | Tensor):
        # Unpack the gradient output if it's a tuple
        if isinstance(grad_output, tuple):
            grad_output, *_ = grad_output

        # Use pop() to ensure we don't use the same activation multiple times
        # and to save memory
        clean = mod_to_clean.pop(module)

        # If noise is tensor, replace acts with noise
        if isinstance(mod_to_noise[module], Tensor):
            # Unsqueeze noise at the sequence dimension
            noise = mod_to_noise[module].unsqueeze(1)
        # If noise is tuple, orthogonally project acts to noise
        elif isinstance(mod_to_noise[module], Tuple):
            noise = (mod_to_noise[module][0].unsqueeze(1)
                     - torch.linalg.vecdot(mod_to_noise[module][0].unsqueeze(1), clean.unsqueeze(1)).unsqueeze(3) * mod_to_noise[module][0].unsqueeze(1)
                     + (mod_to_noise[module][1].unsqueeze(-1) * mod_to_noise[module][0]).unsqueeze(1))

        # Unsqueeze clean at noise batch dimension
        direction = noise - clean.unsqueeze(1)
        # Group heads together if applicable
        if head_dim > 0:
            direction = direction.unflatten(-1, (-1, head_dim))
            grad_output = grad_output.unflatten(-1, (-1, head_dim))

        # Batched dot product
        effect = torch.linalg.vecdot(direction, grad_output.type_as(direction))
        # Save the effect
        name = mod_to_name[module]
        effects[name] = effect

    # Forward hook
    def fwd_hook(module: nn.Module, input: tuple[Tensor] | Tensor, output: tuple[Tensor, ...] | Tensor):
        # Unpack the output if it's a tuple
        if isinstance(output, tuple):
            output, *_ = output

        mod_to_clean[module] = output.detach()

    for name, module in model.named_modules():
        # Hooks need to be able to look up the name of a module
        mod_to_name[module] = name
        # Check if the module is in the paths
        for path, noise in noise_acts.items():
            if not name.endswith(path):
                continue

            # Add a hook to the module
            handles.append(module.register_full_backward_hook(bwd_hook))
            handles.append(module.register_forward_hook(fwd_hook))

            # Save the noise activation
            mod_to_noise[module] = noise

    try:
        yield effects
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()

        # Clear grads on the model just to be safe
        model.zero_grad()

class AttributionDetector(ActivationCovarianceBasedDetector, ABC):
    
    @abstractmethod
    def distance_function(self, effects: Tensor):
        pass
    
    def post_covariance_training(self, **kwargs):
        pass

    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            ablation: str = 'zero',
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None,
            n_pcs: int = 10
            ):
        
        activation_names = [k+'.output' for k in shapes.keys()]

        super().__init__(activation_names, activation_processing_func)
        self.shapes = shapes
        self.n_pcs = n_pcs
        self.output_func = output_func
        self.ablation = ablation

    @torch.enable_grad()
    def train(
        self,
        trusted_data: torch.utils.data.Dataset,
        untrusted_data: torch.utils.data.Dataset | None,
        save_path: Path | str | None,
        batch_size: int = 1,
        **kwargs,
    ):

        assert trusted_data is not None
        dtype = self.model.hf_model.dtype
        device = self.model.hf_model.device

        with torch.no_grad():
            self.noise = self.get_noise_tensor(trusted_data, batch_size, device, dtype)

        # Why shape[-2]? We are going to sum over the last dimension during attribution
        # patching. We'll then use the second-to-last dimension as our main dimension
        # to fit Gaussians to (all earlier dimensions will be summed out first).
        # This is kind of arbitrary and we're putting the onus on the user to make
        # sure this makes sense.

        self._n = 0
        if self.ablation == 'pcs':
            noise_batch_size = self.noise[list(self.noise.keys())[0]][0].shape[0]
        else:
            noise_batch_size = self.noise[list(self.noise.keys())[0]].shape[0]
        
        self._effects = {
            name: torch.zeros(len(trusted_data), noise_batch_size * 32, device=device)
            for name, shape in self.shapes.items()
        }
        self._means = {
            name: torch.zeros(noise_batch_size * 32, device=device)
            for name, shape in self.shapes.items()
        }
        self._Cs = {
            name: torch.zeros(noise_batch_size * 32, noise_batch_size * 32, device=device)
            for name, shape in self.shapes.items()
        }

        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size)

        for i, batch in tqdm(enumerate(dataloader)):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=128) as effects:
                out = self.model(inputs).logits
                out = self.output_func(out)
                # assert out.shape == (batch_size,), out.shape
                out.backward()

            self._n += batch_size

            for name, effect in effects.items():
                # Get the effect at the last token
                effect = effect[:, :, -1]
                # Merge the last dimensions
                effect = effect.reshape(batch_size, -1)
                self._effects[name][i] = effect
                self._means[name], self._Cs[name], _ = (
                    detectors.statistical.helpers.update_covariance(
                        self._means[name], self._Cs[name], self._n, effect
                    )
                )

        self.post_train(untrusted_data)

    def get_noise_tensor(self, trusted_data, batch_size, device, dtype, 
                         subset_size=1000, activation_batch_size=16):
        if self.ablation == 'mean':
            indices = torch.randperm(len(trusted_data))[:subset_size]
            subset = HuggingfaceDataset(
                trusted_data.hf_dataset.select(indices),
                text_key=trusted_data.text_key,
                label_key=trusted_data.label_key
            )

            super().train(subset, None, batch_size=activation_batch_size)
            return {k.replace('.output', ''): v.unsqueeze(0) for k, v in self.means.items()}

        elif self.ablation == 'pcs':
            super().train(trusted_data, None, batch_size=activation_batch_size)
            pcs = {}
            for k, C in self.covariances.items():
                eigenvalues, eigenvectors = torch.linalg.eigh(C)
                sorted_indices = eigenvalues.argsort(descending=True)
                principal_components = eigenvectors[:, sorted_indices[:self.n_pcs]]
                principal_components /= torch.norm(principal_components, dim=0)
                mean_activations = torch.matmul(principal_components.T, self._means[k])
                pcs[k.replace('.output', '')] = (principal_components.T, mean_activations)
            return pcs

        elif self.ablation == 'zero':
            return {
                name: torch.zeros((batch_size, 1, *shape), device=device, 
                                  dtype=dtype)
                for name, shape in self.shapes.items()
            }

    def layerwise_scores(self, batch):
        inputs = utils.inputs_from_batch(batch)
        batch_size = len(inputs)
        # AnomalyDetector.eval() wraps everything in a no_grad block, need to undo that.
        with torch.enable_grad():
            with atp(self.model, self.noise, head_dim=128) as effects:
                out = self.model(inputs).logits
                out = self.output_func(out)
                # assert out.shape == (batch_size,), out.shape
                out.backward()
                # self.sample_grad_func(inputs)

        for name, effect in effects.items():
            effects[name] = effect[:, :, -1].reshape(batch_size, -1)

        distances = self.distance_function(
            effects)    
 
        return distances

    def post_train(self, untrusted_data=None, batch_size=1):
        pass

class MahaAttributionDetector(AttributionDetector):
    def post_train(self, **kwargs):
        self.att_means = self._means
        self.att_covariances = {k: C / (self._n - 1) for k, C in self._Cs.items()}
        if any(torch.count_nonzero(C) == 0 for C in self.att_covariances.values()):
            raise RuntimeError("All zero covariance matrix detected.")

        self.att_inv_covariances = {
            k: detectors.statistical.mahalanobis_detector._pinv(C, rcond=1e-5)
            for k, C in self.att_covariances.items()
        }
        self.att_inv_diag_covariances = {
            k: torch.where(torch.diag(C) > 1.e-5, 1 / torch.diag(C), 0)
            for k, C in self.att_covariances.items()
        }

    def distance_function(self, effects):
        return detectors.statistical.helpers.mahalanobis(
            effects,
            self.att_means,
            self.att_inv_covariances,
            self.att_inv_diag_covariances
        )

    def _get_trained_variables(self, saving: bool = False):
        return{
            "means": self.att_means,
            "inv_covariances": self.att_inv_covariances,
            "inv_diag_covariances": self.att_inv_diag_covariances,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self.att_means = variables["means"]
        self.att_inv_covariances = variables["inv_covariances"]
        self.noise = variables["noise"]
        self.att_inv_diag_covariances = variables["inv_diag_covariances"]


class LOFAttributionDetector(AttributionDetector):
    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            k: int,
            ablation: str = 'mean',
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None
            ):
        super().__init__(shapes, output_func, ablation, activation_processing_func)
        self.k = k

    def post_train(self, **kwargs):
        self.effects = self._effects

    def distance_function(self, test_effects):
        return detectors.statistical.helpers.local_outlier_factor(
            test_effects,
            self.effects,
            self.k
        )

    def _get_trained_variables(self, saving: bool = False):
        return{
            "effects": self.effects,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self.effects = variables["effects"]
        self.noise = variables["noise"]

class IsoForestAttributionDetector(AttributionDetector):

    def post_train(self, **kwargs):
        self.isoforest = {name: IsolationForest().fit(layer_effect.cpu().numpy()) for name, layer_effect in self._effects.items()}

    def distance_function(self, test_effects):
        distances: dict[str, torch.Tensor] = {}

        for name, layer_effects in test_effects.items():

            distances[name] = -self.isoforest[name].decision_function(layer_effects.cpu().numpy())
        
        return distances

    def _get_trained_variables(self, saving: bool = False):
        return{
            "isoforest": self.isoforest,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self.isoforest = variables["isoforest"]
        self.noise = variables["noise"]

class ContrastProbeAttributionDetector(AttributionDetector):
    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            k: int,
            ablation: str = 'mean',
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None
            ):
        super().__init__(shapes, output_func, ablation, activation_processing_func)
        self.k = k

    def _get_trained_variables(self, saving: bool = False):
        return{
            "classifier": self.classifier
        }
    
    def _set_trained_variables(self, variables):
        self.classifier = variables["classifier"]

class QueAttributionDetector(AttributionDetector):

    def post_train(self, untrusted_data, batch_size=1, rcond=1e-5):

        whitening_matrices = {}
        for k, cov in self._Cs.items():
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
        
        data_loader = DataLoader(untrusted_data, batch_size=batch_size, shuffle=False)

        self.untrusted_covariances = {k: torch.zeros(32, 32, device=self.model.device) for k in self.shapes.keys()}
        self._n = 0
        self._effect_means = {k: torch.zeros(32, device=self.model.device) for k in self.shapes.keys()}

        for batch in tqdm(data_loader):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=128) as untrusted_effects:
                out = self.model(inputs).logits
                out = self.output_func(out)
                # assert out.shape == (batch_size,), out.shape
                out.backward()
            
            self._n += batch_size

            for name, effect in untrusted_effects.items():
                # Get the effect at the last token
                effect = effect[:, :, -1]
                # Merge the last dimensions
                effect = effect.reshape(batch_size, -1)
                self._effect_means[name], self.untrusted_covariances[name], _ = detectors.statistical.helpers.update_covariance(
                    self._effect_means[name], self.untrusted_covariances[name], self._n, effect
                    )

        whitened_effects = {
            k: torch.einsum(
                "bi,ij->bj",
                self._effects[k].flatten(start_dim=1) - self._effect_means[k],
                self.whitening_matrices[k],
            )
            for k in self._effects.keys()
        }
        whitened_effects = {
            k: whitened_effects[k].flatten(start_dim=1) - 
            whitened_effects[k].flatten(start_dim=1).mean(dim=0, keepdim=True) 
            for k in whitened_effects.keys()
        }

        self.untrusted_covariances = {k: whitened_effects[k].mT @ whitened_effects[k] for k in whitened_effects.keys()}

    def distance_function(self, test_effects):
        
        whitened_test_effects = {
            k: torch.einsum(
                "bi,ij->bj",
                test_effects[k].flatten(start_dim=1) - self._effect_means[k],
                self.whitening_matrices[k],
            )
            for k in test_effects.keys()
        }

        return detectors.statistical.helpers.quantum_entropy(
            whitened_test_effects,
            batch_covariance = self.untrusted_covariances
        )

    def _get_trained_variables(self, saving: bool = False):
        return {
            "_effect_means": self._effect_means,
            "whitening_matrices": self.whitening_matrices,
            "untrusted_covariances": self.untrusted_covariances,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self._effect_means = variables["_effect_means"]
        self.whitening_matrices = variables["whitening_matrices"]
        self.untrusted_covariances = variables["untrusted_covariances"]
        self.noise = variables["noise"]