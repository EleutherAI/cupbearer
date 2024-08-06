from typing import Any, Callable, Dict, Tuple
from pathlib import Path
import torch
from torch import nn
import pdb

from cupbearer.utils.get_attribution_effects import get_effects
from .activation_extractor import ActivationExtractor
from .core import FeatureExtractor, FeatureCache
from cupbearer.data import HuggingfaceDataset
from cupbearer.utils import guess_device_dtype_from_model


class AttributionEffectExtractor(FeatureExtractor):
    def __init__(
        self,
        names: list[str],
        output_func: Callable[[torch.Tensor], torch.Tensor],
        effect_capture_args: Dict[str, Any],
        head_dim: int = 0,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        trusted_data: HuggingfaceDataset | None = None,
        model: nn.Module | None = None,
        cache_path: str | None = None,
        cache: FeatureCache | None = None,
    ):

        super().__init__(
            feature_names=names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache
        )
        self.activation_names = names
        self.output_func = output_func
        self.head_dim = effect_capture_args['head_dim'] if 'head_dim' in effect_capture_args else None
        self.effect_capture_args = effect_capture_args
        self.cache_path = cache_path
        if effect_capture_args['ablation'] in ['mean', 'pcs']:
            assert (trusted_data is not None) and (model is not None), "Trusted data and model must be provided for mean and PCS ablation"
        
        self.set_model(model)

        self.noise = self.get_noise_tensor(trusted_data)

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        effects = get_effects(
            inputs,
            model=self.model,
            noise_acts=self.noise,
            effect_capture_args=self.effect_capture_args,
            output_func=self.output_func,
            head_dim=self.head_dim
        )
        # Process effects if needed
        for name, effect in effects.items():
            if isinstance(effect, list):
                effect = torch.cat(effect, dim=-1)

        return effects
    

    def get_noise_tensor(self, trusted_data, 
                         max_steps=75, activation_batch_size=16):
        from cupbearer.detectors.statistical import MahalanobisDetector

        if self.effect_capture_args['ablation'] in ['mean', 'pcs']:

            if self.cache_path is not None:
                cache = (FeatureCache.load(self.cache_path, 
                                          device=guess_device_dtype_from_model(self.model)[0]) 
                                          if Path(self.cache_path).exists()
                                          else FeatureCache(device=guess_device_dtype_from_model(self.model)[0]))
            else:
                cache = None

            extractor = ActivationExtractor(
                names=list(map(lambda x: x+'.output', self.activation_names)),
                individual_processing_fn=self.individual_processing_fn,
                cache=cache
            )
            maha_detector = MahalanobisDetector(
                feature_extractor=extractor,
            )
            maha_detector.set_model(self.model)

            maha_detector.train(trusted_data, None, batch_size=activation_batch_size, max_steps=max_steps)
            if cache is not None:
                cache.store(self.cache_path, overwrite=True)
            means = maha_detector.means

            if self.effect_capture_args['ablation'] == 'mean':
                return {k.replace('.output', ''): v for k, v in means['trusted'].items()}

            elif self.effect_capture_args['ablation'] == 'pcs':
                covariances = maha_detector.covariances['trusted']
                means = maha_detector.means['trusted']
                pcs = {}
                for k, C in covariances.items():
                    eigenvalues, eigenvectors = torch.linalg.eigh(C)
                    sorted_indices = eigenvalues.argsort(descending=True)
                    principal_components = eigenvectors[:, sorted_indices[:self.effect_capture_args['n_pcs']]]
                    principal_components /= torch.norm(principal_components, dim=0)
                    mean_activations = torch.matmul(principal_components.T, means[k])
                    pcs[k.replace('.output', '')] = (principal_components.T, mean_activations)

                return pcs

        elif self.effect_capture_args['ablation'] in ['zero', 'grad_norm']:
            return {k: None for k in self.activation_names}
        
        else:
            raise ValueError(f"Unknown ablation method: {self.effect_capture_args['ablation']}")
