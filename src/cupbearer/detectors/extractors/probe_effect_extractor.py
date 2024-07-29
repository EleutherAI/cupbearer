from typing import Any, Callable, Dict
import torch
from torch import nn
import pdb

from cupbearer.utils.get_attribution_effects import get_effects, prepare_model_for_effects
from cupbearer.utils.get_activations import get_activations
from cupbearer.utils.classifier import Classifier
from .core import FeatureExtractor, FeatureCache
from .attribution_effect_extractor import AttributionEffectExtractor
from cupbearer.data import HuggingfaceDataset
from cupbearer.utils import guess_device_dtype_from_model, inputs_from_batch

class ProbeEffectExtractor(FeatureExtractor):
    def __init__(
        self,
        probe_layers: list[str],
        intervention_layers: list[str],
        output_func: Callable[[torch.Tensor], torch.Tensor],
        effect_capture_args: Dict[str, Any],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        trusted_data: HuggingfaceDataset | None = None,
        model: nn.Module | None = None,
        cache: FeatureCache | None = None,
        cache_path: str | None = None,
    ):
        def extract_layer_number(layer_name):
            return int(''.join(filter(str.isdigit, layer_name)))

        min_intervention_layer = min(extract_layer_number(layer) for layer in intervention_layers)
        self.probe_layers = [layer for layer in probe_layers if extract_layer_number(layer) > min_intervention_layer]
        self.intervention_layers = intervention_layers

        super().__init__(
            feature_names=self.probe_layers,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache
        )

        self.activation_names = intervention_layers
        self.output_func = output_func
        self.effect_capture_args = effect_capture_args
        self.cache_path = cache_path
        
        self.set_model(model)
        self.probes = self.train_probes(trusted_data)
        self.noise = AttributionEffectExtractor.get_noise_tensor(self, trusted_data)

    def train_probes(self, trusted_data):
        device, dtype = guess_device_dtype_from_model(self.model)
        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=8)
        
        activations = {layer: [] for layer in self.probe_layers}
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                _, batch_labels = batch
                inputs = inputs_from_batch(batch)
                batch_activations = get_activations(inputs, model = self.model, names = [l + '.output' for l in self.probe_layers])
                for layer in self.probe_layers:
                    activations[layer].append(self.individual_processing_fn(batch_activations[layer + '.output'].cpu(), inputs, layer))
                labels.append(batch_labels)

        activations = {layer: torch.cat(activations[layer], dim=0) for layer in self.probe_layers}
        labels = torch.cat(labels, dim=0)
        probes = {}
        for layer in self.probe_layers:
            probes[layer] = Classifier(input_dim=activations[layer].shape[-1], device=device, dtype=dtype)
            probes[layer].fit_cv(activations[layer].to(device), labels.to(device))

        return probes

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        
        effects = get_effects(
            inputs,
            model=self.model,
            noise_acts=self.noise,
            effect_capture_args=self.effect_capture_args,
            output_func=self.output_func,
            head_dim=self.effect_capture_args.get('head_dim', None))

        features = {layer: [] for layer in self.probe_layers}
        with torch.enable_grad():
            for layer in self.probe_layers:
                with prepare_model_for_effects(self.model, 
                                               self.noise, 
                                               self.effect_capture_args,
                                               self.effect_capture_args.get('head_dim', None)) as probe_effects:
                    activations = get_activations(inputs, model = self.model, names = [l + '.output' for l in self.probe_layers], enable_grad=True)
                    layer_activation = activations[layer + '.output']
                    processed_activation = self.individual_processing_fn(layer_activation, inputs, layer)
                    probe_out = self.probes[layer](processed_activation)
                    probe_out.sum().backward(retain_graph=True)

                for intervention_layer in probe_effects:
                    features[layer].append(probe_effects[intervention_layer] - effects[intervention_layer])

                features[layer] = torch.cat(features[layer], dim=-1)

                self.model.zero_grad()


        return features