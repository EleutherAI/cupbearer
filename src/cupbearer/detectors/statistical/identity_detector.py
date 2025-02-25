import torch
from cupbearer.detectors.anomaly_detector import AnomalyDetector


class IdentityDetector(AnomalyDetector):
    """
    Detector that simply returns the features as anomaly scores.
    Useful for features that are already anomaly scores.
    """
    def __init__(self, feature_extractor, reduction="none", **kwargs):
        """
        Args:
            feature_extractor: The feature extractor to use
            reduction: How to reduce features beyond the first dimension.
                Options: "none" (default), "mean", "sum", "max", "min"
        """
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor
        self.reduction = reduction

    def _train(self, trusted_dataloader, untrusted_dataloader, **kwargs):
        # No training required for identity detector
        pass

    def _compute_layerwise_scores(self, inputs, features):
        scores = {}
        for name, feature in features.items():
            # Apply reduction if needed
            if self.reduction == "mean" and feature.dim() > 1:
                # Mean over all dimensions except the first (batch dimension)
                feature = feature.mean(dim=tuple(range(1, feature.dim())))
            elif self.reduction == "sum" and feature.dim() > 1:
                feature = feature.sum(dim=tuple(range(1, feature.dim())))
            elif self.reduction == "max" and feature.dim() > 1:
                feature = feature.max(dim=tuple(range(1, feature.dim())))[0]
            elif self.reduction == "min" and feature.dim() > 1:
                feature = feature.min(dim=tuple(range(1, feature.dim())))[0]
            
            scores[name] = feature
        
        return scores

    def _get_trained_variables(self):
        return {}

    def _set_trained_variables(self, variables):
        pass 