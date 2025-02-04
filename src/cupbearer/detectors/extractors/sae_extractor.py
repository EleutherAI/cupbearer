from typing import Any, Callable, Dict, List, Literal
import torch

from cupbearer import utils
from .core import FeatureExtractor
from sae import Sae

class SaeExtractor(FeatureExtractor):
    def __init__(
        self,
        layers: List[int],
        names: List[str],
        hookpoint_type: Literal["mlp", "residual"] = "mlp",
        sae_model: str = "EleutherAI/sae-llama-3.1-8b-64x",
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor] | None = None,
        global_processing_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ):
        self.layers = layers
        self.hookpoint_type = hookpoint_type
        self.sae_model = sae_model
        self.names = names
        # Generate hookpoint names
        self.hookpoint_names = self._generate_hookpoint_names()
        self.higher_individual_processing_fn = individual_processing_fn
        
        super().__init__(
            feature_names=self.names,
            individual_processing_fn=None,
            global_processing_fn=global_processing_fn,
        )
        
        # Load SAEs
        self.saes = self._load_saes()

    def _generate_hookpoint_names(self) -> List[str]:
        if self.hookpoint_type == "mlp":
            return [f"layers.{layer}.mlp" for layer in self.layers]
        elif self.hookpoint_type == "residual":
            return [f"layers.{layer}" for layer in self.layers]
        else:
            raise ValueError(f"Invalid hookpoint_type: {self.hookpoint_type}")

    def _load_saes(self) -> Dict[str, Sae]:
        saes = {}
        available_hookpoints = set(['layers.23.mlp', 'layers.29.mlp', 'layers.23', 'layers.29'])
        
        for name, hookpoint_name in zip(self.names, self.hookpoint_names):
            if hookpoint_name not in available_hookpoints:
                raise ValueError(f"Hookpoint '{hookpoint_name}' is not available in the SAE model.")
            saes[name] = Sae.load_from_hub(self.sae_model, hookpoint=hookpoint_name)
        
        return saes

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        activations = utils.get_activations(inputs, model=self.model, names=self.names)
        
        features = {}
        for name, activation in activations.items():
            sae = self.saes[name].to(activation.device)
            act = self.higher_individual_processing_fn(activation, inputs, name)
            feat = sae.encode(act)

            latent_shape = self.saes[name].W_dec.shape[0]
            buf = torch.zeros(act.shape[:-1] + (latent_shape,)).to(activation.device)
            buf.scatter_(-1, feat.top_indices, feat.top_acts)
            features[name] = buf

        return features