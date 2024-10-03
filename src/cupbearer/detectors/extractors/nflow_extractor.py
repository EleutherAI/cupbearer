from typing import Any, Callable, Dict
import torch
from pathlib import Path

from cupbearer import utils
from .core import FeatureExtractor, FeatureCache
from flows import Flow

class NFlowExtractor(FeatureExtractor):
    def __init__(
        self,
        names: list[str],
        flow_paths: Dict[str, Path],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(
            feature_names=names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
        )
        self.names = names
        self.flows = {name: Flow.load_from_disk(path, strict=False) for name, path in zip(names, flow_paths)}

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        activations = utils.get_activations(inputs, model=self.model, names=[n + '.output' for n in self.names])
        
        features = {}
        for name, activation in activations.items():
            flow = self.flows[name.replace('.output', '')].to(activation.device)
            features[name] = flow(activation).z

        return features