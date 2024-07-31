from abc import ABC

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Any, Tuple, Dict
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import torch
from cupbearer import detectors, utils
from cupbearer.detectors.statistical.statistical import ActivationCovarianceBasedDetector
from cupbearer.data import HuggingfaceDataset
from cupbearer.models import HuggingfaceLM
from torch import Tensor, nn
from torch.utils.data import DataLoader
from collections import defaultdict
import pdb
import os
import matplotlib.pyplot as plt


@contextmanager
def edge_intervention(model: nn.Module, noise_acts: Dict[str, Tensor], n_layers: int = 4, *, head_dim: int = 0):
    handles: list[nn.modules.module.RemovableHandle] = []
    mod_to_clean: dict[nn.Module, Tensor] = {}
    mod_to_noise: dict[nn.Module, Tensor] = {}
    effects: defaultdict[str, Tensor] = defaultdict(list)
    mod_to_name: dict[nn.Module, str] = {}
    mod_to_parents: dict[nn.Module, list[nn.Module]] = {}
    
    def get_layer(name: str) -> int:
        return next((int(part) for part in name.split('.') if part.isdigit()), float('inf'))
    
    def bwd_hook(module: nn.Module, grad_input: tuple[Tensor, ...] | Tensor, grad_output: tuple[Tensor, ...] | Tensor):
        if isinstance(grad_output, tuple):
            grad_output, *_ = grad_output
        
        target_name = mod_to_name[module]
        head_target = head_dim > 0 and target_name.endswith('self_attn')
        if head_target:
            grad_output = grad_output.unflatten(-1, (-1, head_dim))

        for original_module in mod_to_parents[module]:
            original_name = mod_to_name[original_module]
            clean = mod_to_clean[original_module]
            noise = mod_to_noise[original_module]
            
            if original_name.endswith('self_attn'):
                # Reshape tensors to (..., n_heads, head_dim)
                n_heads = clean.shape[-1] // head_dim
                clean = clean.view(*clean.shape[:-1], n_heads, head_dim)
                noise = noise.view(1, n_heads, head_dim)

                # Compute effects for each head
                effects_list = []
                for i in range(n_heads):
                    direction = (noise[:, i, :] - clean[..., i, :].unsqueeze(1)).unsqueeze(-2).expand_as(clean.unsqueeze(1))
                    if not head_target:
                        direction = direction.flatten(-2, -1)
                        effect = torch.linalg.vecdot(direction, grad_output.type_as(direction)).unsqueeze(-1)
                    else:
                        effect = torch.linalg.vecdot(direction, grad_output.type_as(direction))
                    effects_list.append(effect)
                
                # Concatenate effects
                effect = torch.cat(effects_list, dim=-1).detach()
            else:
                direction = noise - clean.unsqueeze(1)
                effect = torch.linalg.vecdot(direction, grad_output.type_as(direction)).unsqueeze(-1)
            
            effects[original_name].append(effect.clone().detach())

    def fwd_hook(module: nn.Module, input: tuple[Tensor] | Tensor, output: tuple[Tensor, ...] | Tensor):
        if isinstance(output, tuple):
            output, *_ = output
        mod_to_clean[module] = output.detach()

    modules = list(model.named_modules())


    for i, (name, module) in enumerate(modules):
        mod_to_name[module] = name
        mod_to_parents[module] = []
        current_layer = get_layer(name)
        
        for j in range(0, i):
            parent_name, parent_module = modules[j]
            parent_layer = get_layer(parent_name)
            if current_layer - parent_layer <= n_layers and parent_name in noise_acts and (mod_to_name[module].endswith('mlp') or mod_to_name[module].endswith('self_attn')):
                mod_to_parents[module].append(parent_module)
        
        if name in noise_acts:
            handles.append(module.register_forward_hook(fwd_hook))
            mod_to_noise[module] = noise_acts[name]
        
        if mod_to_parents[module]:
            handles.append(module.register_full_backward_hook(bwd_hook))

    try:
        yield effects.clone().detach()

    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()


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

        clean = mod_to_clean.pop(module)
        seq = True
        if clean.ndim == 4:
            seq = False
            clean = clean.view(-1, 1, clean.shape[-1])
        
        if mod_to_noise[module] is None:
            noise = grad_output.unsqueeze(1) + clean.unsqueeze(1)
        # If noise is tensor, replace acts with noise
        elif isinstance(mod_to_noise[module], Tensor):
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
        elif not seq:
            direction = direction.squeeze(-2)
            grad_output = grad_output.view(-1, 1, grad_output.shape[-1])

        # Batched dot product
        effect = torch.linalg.vecdot(direction, grad_output.type_as(direction))
        # Save the effect
        name = mod_to_name[module]
        if name in effects:
            effects[name] += effect.clone().detach()
        else:
            effects[name] = effect.clone().detach()

    # Forward hook
    def fwd_hook(module: nn.Module, input: tuple[Tensor] | Tensor, output: tuple[Tensor, ...] | Tensor):
        # Unpack the output if it's a tuple
        if isinstance(output, tuple):
            output, *_ = output

        mod_to_clean[module] = output.clone().detach()

    for name, module in model.named_modules():
        # Hooks need to be able to look up the name of a module
        mod_to_name[module] = name
        # Check if the module is in the paths
        
        for path, noise in noise_acts.items():
            if not name == path:
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

@contextmanager
def raw_gradient_capture(model: nn.Module, noise: dict[str, torch.Tensor]):
    layers = list(noise.keys())
    handles = []
    effects = {}
    mod_to_name = {}

    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        name = mod_to_name[module]
        effects[name] = grad_output.detach().unsqueeze(1)

    for name, module in model.named_modules():
        mod_to_name[module] = name
        if name in layers:
            handles.append(module.register_full_backward_hook(backward_hook))

    try:
        yield effects
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad()


class AttributionDetector(ActivationCovarianceBasedDetector, ABC):
    
    def post_covariance_training(self, **kwargs):
        pass

    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            effect_capture_method: str,
            effect_capture_args: dict = dict(),
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
            append_activations: bool = False,
            head_dim: int = 128,
            **kwargs
            ):
        
        activation_names = [k+'.output' for k in shapes.keys()]

        super().__init__(activation_names, activation_processing_func, **kwargs)
        self.shapes = shapes
        self.output_func = output_func
        self.effect_capture_method = effect_capture_method
        self.effect_capture_args = effect_capture_args
        self.append_activations = append_activations
        self.head_dim = head_dim

    def _setup_effect_capture(self, noise: dict[str, torch.Tensor] | None = None):

        if self.effect_capture_method == 'atp':
            return lambda model: atp(model, noise, head_dim=self.head_dim)
        elif self.effect_capture_method == 'edge_intervention':
            return lambda model: edge_intervention(model, noise, head_dim=self.head_dim)
        elif self.effect_capture_method == 'raw':
            return lambda model: raw_gradient_capture(model, noise)
        else:
            raise ValueError(f"Unknown effect capture method: {self.effect_capture_method}")


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
        if isinstance(self.model, HuggingfaceLM):
            dtype = self.model.hf_model.dtype
            device = self.model.hf_model.device
        else:
            dtype = next(self.model.parameters()).dtype
            device = next(self.model.parameters()).device


        if self.effect_capture_method in ['atp', 'edge_intervention']:
            with torch.no_grad():
                self.noise = self.get_noise_tensor(trusted_data, batch_size, device, dtype)
        else:
            self.noise = {name: None for name in self.shapes.keys()}

        effect_capture_func = self._setup_effect_capture(self.noise)

        self._n = 0
        self._effects = dict()
        
        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=1)

        sample_batch = next(iter(dataloader))

        # Perform a single forward and backward pass to get the effect shapes
        inputs = utils.inputs_from_batch(sample_batch)
        inputs = utils.inputs_to_device(inputs, device)
        with effect_capture_func(self.model) as sample_effects:
            if isinstance(self.model, HuggingfaceLM):
                out = self.model(inputs).logits
            else:
                out = self.model(inputs)
            out = self.output_func(out)
            out.backward()

        if self.append_activations:
            acts = self.get_activations(sample_batch)

        for name, effect in sample_effects.items():
            if isinstance(self.model, HuggingfaceLM):
                effect = effect[:, :, -1].reshape(1, -1)
            if self.append_activations:
                effect = torch.cat([
                    effect, acts[name + '.output']], dim=-1)
            if isinstance(self.model, HuggingfaceLM):
                self._effects[name] = torch.zeros(
                    len(trusted_data),
                    effect.shape[-1],
                    device=device
                )
            # For vision models, each sample gives us a batch of effects
            else:
                self._effects[name] = torch.zeros(
                    len(trusted_data),
                    torch.tensor(effect.shape[:-1]).to(int).prod().item(),
                    effect.shape[-1],
                    device=device
                )
        self._means = {
            name: torch.zeros(effect.shape[-1], device=device) if isinstance(effect, torch.Tensor)
            else torch.zeros(torch.cat(effect, dim=-1).shape[-1], device=device)
            for name, effect in self._effects.items()
        }
        self._Cs = {
            name: torch.zeros(effect.shape[-1], effect.shape[-1], device=device) if isinstance(effect, torch.Tensor)
            else torch.zeros(torch.cat(effect, dim=-1).shape[-1], torch.cat(effect, dim=-1).shape[-1], device=device)
            for name, effect in self._effects.items()
        }

        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size)

        for i, batch in tqdm(enumerate(dataloader)):
            inputs = utils.inputs_from_batch(batch)
            inputs = utils.inputs_to_device(inputs, device)
            with effect_capture_func(self.model) as effects:
                out = self.model(inputs)
                if isinstance(self.model, HuggingfaceLM):
                    out = out.logits
                out = self.output_func(out)
                out.backward()

            self._n += batch_size
            if self.append_activations:
                acts = self.get_activations(batch)          
                for name, act in acts.items():
                    effects[name.replace('.output', '')] = torch.cat([
                        effects[name.replace('.output', '')][:, :, -1].reshape(1, -1), act], dim=-1
                    )

            for name, effect in effects.items():
                if isinstance(effect, list):
                    effect = torch.cat(effect, dim=-1)
                # Get the effect at the last token
                if not self.append_activations and isinstance(self.model, HuggingfaceLM):
                    effect = effect[:, :, -1].reshape(batch_size, -1)
                for j in range(batch_size):
                    self._effects[name][i * batch_size + j] = effect.view(batch_size, -1, *effect.shape[1:])[j]
                self._means[name], self._Cs[name], _ = (
                    detectors.statistical.helpers.update_covariance(
                        self._means[name], self._Cs[name], self._n, effect
                    )
                )

        self.post_train(untrusted_data=untrusted_data)

    def get_noise_tensor(self, trusted_data, batch_size, device, dtype, 
                         subset_size=1000, activation_batch_size=8):
        if self.effect_capture_args['ablation'] == 'grad_norm':
            return {k: None for k in self.shapes.keys()}
        
        elif self.effect_capture_args['ablation'] == 'mean':
            indices = torch.randperm(len(trusted_data))[:subset_size]
            subset = HuggingfaceDataset(
                trusted_data.hf_dataset.select(indices),
                text_key=trusted_data.text_key,
                label_key=trusted_data.label_key
            )

            super().train(subset, None, batch_size=activation_batch_size)
            return {k.replace('.output', ''): v.unsqueeze(0) for k, v in self.means.items()}

        elif self.effect_capture_args['ablation'] == 'pcs':
            super().train(trusted_data, None, batch_size=activation_batch_size)
            pcs = {}
            for k, C in self.covariances.items():
                eigenvalues, eigenvectors = torch.linalg.eigh(C)
                sorted_indices = eigenvalues.argsort(descending=True)
                principal_components = eigenvectors[:, sorted_indices[:self.effect_capture_args['n_pcs']]]
                principal_components /= torch.norm(principal_components, dim=0)
                mean_activations = torch.matmul(principal_components.T, self._means[k])
                pcs[k.replace('.output', '')] = (principal_components.T, mean_activations)
            return pcs

        elif self.effect_capture_args['ablation'] == 'zero':
            return {
                name: torch.zeros((batch_size, 1, *shape), device=device, 
                                  dtype=dtype)
                for name, shape in self.shapes.items()
            }

    def layerwise_scores(self, batch):

        effect_capture_func = self._setup_effect_capture(self.noise)

        inputs = utils.inputs_from_batch(batch)
        batch_size = len(inputs)
        # AnomalyDetector.eval() wraps everything in a no_grad block, need to undo that.
        with torch.enable_grad():
            with effect_capture_func(self.model) as effects:
                if isinstance(self.model, HuggingfaceLM):
                    out = self.model(inputs).logits
                else:
                    out = self.model(inputs)
                out = self.output_func(out)
                out.backward()

        if self.append_activations:
            acts = self.get_activations(batch)          
            for name, act in acts.items():
                effects[name.replace('.output', '')] = torch.cat([
                    effects[name.replace('.output', '')][:, :, -1].reshape(1, -1), act], dim=-1
                )

        seq = False
        for name, effect in effects.items():
            if isinstance(effect, list):
                effect = torch.cat(effect, dim=-1)
            if isinstance(self.model, HuggingfaceLM):
                if not self.append_activations:
                    effects[name] = effect[:, :, -1].reshape(batch_size, -1)
                seq = True

        if seq:
            scores = {
                    k: self._individual_layerwise_score(k, v)
                    for k, v in effects.items()
                }
        else:
            scores = {
                k: self._individual_layerwise_score(k, v).reshape(batch_size, -1).mean(dim=-1)
                for k, v in effects.items()
            }
 
        return scores

    def post_train(self, untrusted_data=None, batch_size=1):
        pass

class ImpactfulDeviationDetector(AttributionDetector):
    def __init__(
            self, 
            shapes: dict[str, tuple[int, ...]], 
            output_func: Callable[[torch.Tensor], torch.Tensor],
            effect_capture_method: str,
            effect_capture_args: dict = dict(),
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
            impact_threshold: float = 0.05,
            **kwargs
            ):
        super().__init__(shapes, output_func, effect_capture_method, effect_capture_args, activation_processing_func, **kwargs)
        self.impact_threshold = impact_threshold
        self.layer_aggregation = 'sum'

    def post_train(self, untrusted_data=None, batch_size=1):
        self.mean_impacts = {name: effect.abs().mean(dim=0) for name, effect in self._effects.items()}
        self.low_impact_masks = {name: self.get_low_impact_mask(name) for name in self.mean_impacts.keys()}

    def get_low_impact_mask(self, name: str):
        threshold = torch.quantile(self.mean_impacts[name], self.impact_threshold)
        return self.mean_impacts[name] < threshold

    def save_impact_histograms(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        for name, impacts in self.mean_impacts.items():
            plt.figure(figsize=(10, 6))
            plt.hist(impacts.cpu().numpy(), bins=50, edgecolor='black')
            plt.title(f'Impact Histogram for {name}')
            plt.xlabel('Impact')
            plt.ylabel('Frequency')
            plt.axvline(x=self.impact_threshold, color='r', linestyle='--', label='Threshold')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{name}_impact_histogram.png'))
            plt.close()

    def train(self, trusted_data, untrusted_data, save_path, batch_size=1, **kwargs):
        super().train(trusted_data, untrusted_data, save_path, batch_size, **kwargs)
        
        # Save impact histograms after training
        if save_path:
            histogram_dir = os.path.join(save_path, 'impact_histograms')
            self.save_impact_histograms(histogram_dir)

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        # Focus on units that had low impact in trusted data but high impact in this sample
        scores = effects.abs() * self.low_impact_masks[name].float()

        return scores.sum(dim=-1)

    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self._means,
            "low_impact_masks": self.low_impact_masks,
            "mean_impacts": self.mean_impacts,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self._means = variables["means"]
        self.mean_impacts = variables["mean_impacts"]
        self.low_impact_masks = {name: self.get_low_impact_mask(name) for name in self.mean_impacts.keys()}
        self.noise = variables["noise"]

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

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        return detectors.statistical.helpers.mahalanobis(
            effects,
            self.att_means[name],
            self.att_inv_covariances[name]
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

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        return detectors.statistical.helpers.local_outlier_factor(
            effects,
            self.effects[name],
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

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        return -self.isoforest[name].decision_function(effects.cpu().numpy())
        

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

        if isinstance(self.model, HuggingfaceLM):
            dtype = self.model.hf_model.dtype
            device = self.model.hf_model.device
        else:
            dtype = next(self.model.parameters()).dtype
            device = next(self.model.parameters()).device

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

        self.untrusted_covariances = {k: torch.zeros_like(self._Cs[k]) for k in self._Cs.keys()}
        self._n = 0
        self._untrusted_effect_means = {k: torch.zeros_like(self._means[k]) for k in self._means.keys()}
        self._untrusted_effects = {k: torch.zeros((len(untrusted_data), *self._effects[k].shape[1:]), device=device) for k in self._effects.keys()}

        for i, batch in tqdm(enumerate(data_loader)):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=self.head_dim) as untrusted_effects:
                if isinstance(self.model, HuggingfaceLM):
                    out = self.model(inputs).logits
                else:
                    out = self.model(inputs)
                out = self.output_func(out)
                # assert out.shape == (batch_size,), out.shape
                out.backward()

            
            self._n += batch_size

            if self.append_activations:
                acts = self.get_activations(batch)

            for name, effect in untrusted_effects.items():
                # Get the effect at the last token
                if isinstance(self.model, HuggingfaceLM):
                    if self.append_activations:
                        effect = torch.cat([effect[:, :, -1].reshape(1, -1), acts[name + '.output']], dim=-1)
                    else:
                        effect = effect[:, :, -1].reshape(batch_size, -1)
                # Merge the last dimensions
                self._untrusted_effect_means[name], self.untrusted_covariances[name], _ = detectors.statistical.helpers.update_covariance(
                    self._untrusted_effect_means[name], self.untrusted_covariances[name], self._n, effect
                    )
                for j in range(batch_size):
                    self._untrusted_effects[name][i * batch_size + j] = effect.view(batch_size, -1, *effect.shape[1:])[j]

        # Center and whiten effects
        whitened_effects = {
            k: torch.einsum(
                "bi,ij->bj",
                self._untrusted_effects[k].flatten(end_dim=-2) - self._means[k],
                self.whitening_matrices[k],
            )
            for k in self._effects.keys()
        }

        whitened_effects = {
            k: whitened_effects[k] - whitened_effects[k].mean(dim=0, keepdim=True)
            for k in whitened_effects.keys()
        }

        self.untrusted_covariances = {k: whitened_effects[k].mT @ whitened_effects[k] for k in whitened_effects.keys()}

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        whitened_test_effects = torch.einsum(
            "bi,ij->bj",
            effects.flatten(start_dim=1) - self._untrusted_effect_means[name],
            self.whitening_matrices[name],
        )

        return detectors.statistical.helpers.quantum_entropy(
            whitened_test_effects,
            batch_cov = self.untrusted_covariances[name]
        )

    def _get_trained_variables(self, saving: bool = False):
        return {
            "_effect_means": self._untrusted_effect_means,
            "whitening_matrices": self.whitening_matrices,
            "untrusted_covariances": self.untrusted_covariances,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self._untrusted_effect_means = variables["_effect_means"]
        self.whitening_matrices = variables["whitening_matrices"]
        self.untrusted_covariances = variables["untrusted_covariances"]
        self.noise = variables["noise"]