import torch
from cupbearer.detectors.activation_based import ActivationBasedDetector
from einops import rearrange

class LaplaceDetector(ActivationBasedDetector):
    def _train(self, *args, **kwargs):
        pass

    def _compute_layerwise_scores(self, inputs, features):
        batch_size = next(iter(features.values())).shape[0]
        features = {
            k: rearrange(v, "batch ... dim -> (batch ...) dim")
            for k, v in features.items()
        }
        d = torch.distributions.Laplace(0, 1)
        # Calculate Laplace density for Laplace(0,1)
        scores = {
            k: d.log_prob(v).sum(dim=-1).to(torch.float32) for k, v in features.items()
        }
        
        return scores