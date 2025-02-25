from datasets import Dataset
import torch
from tyche import VolumeConfig, VolumeEstimator

from .core import FeatureExtractor


class BasinVolumeExtractor(FeatureExtractor):
    """
    Extractor that computes the basin volume for each text input using the VolumeEstimator.
    """
    def __init__(
        self,
        n_samples=10,
        cutoff=1e-2,
        max_seq_len=1024,
        **kwargs
    ):
        super().__init__(feature_names=["basin_volume"], **kwargs)
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.max_seq_len = max_seq_len

    def compute_features(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dataset = Dataset.from_dict({"text": list(inputs)})
        cfg = VolumeConfig(
            model=self.model,
            tokenizer=self.model.tokenizer,
            dataset=dataset,
            text_key="text",
            n_samples=self.n_samples,
            cutoff=self.cutoff,
            max_seq_len=self.max_seq_len,
            val_size=len(inputs),
            cache_mode=None,
            chunking=False,
            model_type="causal",
            implicit_vectors=True,
            reduction=None
        )
        estimator = VolumeEstimator.from_config(cfg)
        with torch.no_grad():
            result = estimator.run()
        
        # Transpose the result to have shape [batch_size, n_samples]
        volume = result.estimates.transpose(0, 1)
        
        return {"basin_volume": volume}