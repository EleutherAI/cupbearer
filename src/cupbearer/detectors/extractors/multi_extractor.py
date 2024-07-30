from typing import List, Dict, Any, Callable
import torch
from collections import defaultdict
import pdb

from .core import FeatureExtractor, FeatureCache

class MultiExtractor(FeatureExtractor):
    def __init__(
        self,
        extractors: List[FeatureExtractor],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        cache: FeatureCache | None = None,
        feature_groups: Dict[str, List[str]] | None = None,
    ):
        feature_names = feature_groups.keys()
        
        super().__init__(
            feature_names=feature_names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache
        )
        
        self.extractors = extractors
        self.feature_groups = feature_groups

    def set_model(self, model: torch.nn.Module):
        self.model = model
        for extractor in self.extractors:
            extractor.set_model(model)

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        grouped_features = defaultdict(list)
        for extractor in self.extractors:
            features = extractor.compute_features(inputs)
            for name, feature in features.items():
                for group_name in self.feature_groups.keys():
                    if name in self.feature_groups[group_name]:
                        grouped_features[group_name].append(feature)
        grouped_features = {group_name: torch.cat(feature_list, dim=-1) for group_name, feature_list in grouped_features.items()}
                
        return grouped_features