from typing import Any, Callable, Dict, Tuple

import torch
from torch import nn

from cupbearer.utils.get_attribution_effects import get_effects
from .core import FeatureExtractor
from cupbearer.data import HuggingfaceDataset


class AttributionEffectExtractor(FeatureExtractor):
    def __init__(
        self,
        activation_names: list[str],
        output_func: Callable[[torch.Tensor], torch.Tensor],
        effect_capture_args: Dict[str, Any],
        head_dim: int = 0,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        trusted_data: HuggingfaceDataset | None = None,
        model: nn.Module | None = None,
    ):

        super().__init__(
            feature_names=activation_names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
        )
        self.activation_names = activation_names
        self.output_func = output_func
        self.head_dim = head_dim
        self.effect_capture_args = effect_capture_args
        
        if effect_capture_args['ablation'] in ['mean', 'pcs']:
            assert (trusted_data is not None) and (model is not None), "Trusted data and model must be provided for mean and PCS ablation"
        
        self.set_model(model)

        self.noise = self.get_noise_tensor(trusted_data)


    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        effects = get_effects(
            inputs,
            self.model,
            self.noise,
            self.effect_capture_args,
            output_func=self.output_func,
            head_dim=self.head_dim
        )

        # Process effects if needed
        for name, effect in effects.items():
            if isinstance(effect, list):
                effect = torch.cat(effect, dim=-1)
            if effect.ndim > 2:  # For language models, take the last token
                effects[name] = effect[:, :, -1].reshape(len(inputs), -1)

        return effects
    

    def get_noise_tensor(self, trusted_data, 
                         subset_size=1000, activation_batch_size=8):
        from cupbearer.detectors.statistical import MahalanobisDetector

        if self.effect_capture_args['ablation'] in ['mean', 'pcs']:
            maha_detector = MahalanobisDetector(
                activation_names=list(map(lambda x: x+ '.output', self.activation_names)),
                individual_processing_fn=self.individual_processing_fn,
                global_processing_fn=self.global_processing_fn,
            )
            maha_detector.set_model(self.model)
            indices = torch.randperm(len(trusted_data))[:subset_size]
            subset = HuggingfaceDataset(
                trusted_data.hf_dataset.select(indices),
                text_key=trusted_data.text_key,
                label_key=trusted_data.label_key
            )

            maha_detector.train(subset, None, batch_size=activation_batch_size)
            means = maha_detector.means

            if self.effect_capture_args['ablation'] == 'mean':
                return {k.replace('.output', ''): v for k, v in means.items()}

            elif self.effect_capture_args['ablation'] == 'pcs':
                covariances = maha_detector.covariances
                means = maha_detector.means
                pcs = {}
                for k, C in covariances.items():
                    eigenvalues, eigenvectors = torch.linalg.eigh(C)
                    sorted_indices = eigenvalues.argsort(descending=True)
                    principal_components = eigenvectors[:, sorted_indices[:self.effect_capture_args['n_pcs']]]
                    principal_components /= torch.norm(principal_components, dim=0)
                    mean_activations = torch.matmul(principal_components.T, means[k])
                    pcs[k.replace('.output', '')] = (principal_components.T, mean_activations)

                return pcs

        elif self.effect_capture_args['ablation'] == 'zero':
            return None
        
        else:
            raise ValueError(f"Unknown ablation method: {self.effect_capture_args['ablation']}")
