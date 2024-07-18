import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer import utils
from cupbearer.data import MixedData

from .extractors import FeatureExtractor


class AnomalyDetector(ABC):
    """Base class for model-based anomaly detectors.

    These are the main detectors that users will interact with directly.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        layer_aggregation: str = "mean",
    ):
        self.feature_extractor = feature_extractor
        self.layer_aggregation = layer_aggregation

    @abstractmethod
    def _compute_layerwise_scores(
        self, inputs: Any, features: Any
    ) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        Each element of the returned dictionary must have shape (batch_size, ).

        If a detector can't compute layerwise scores, it should instead return
        a dictionary with only one element (by convention using an 'all' key).
        """
        pass

    @abstractmethod
    def _train(
        self,
        trusted_dataloader: DataLoader | None,
        untrusted_dataloader: DataLoader | None,
        **kwargs,
    ):
        """Train the anomaly detector with the given datasets on the given model.

        At least one of trusted_dataloader or untrusted_dataloader will be provided.

        The dataloaders return tuples (batch, features), where `batch` will be created
        directly from the underlying dataset (so potentially include labels) and
        `features` is None or the output of the feature extractor.
        """

    def _get_trained_variables(self):
        return {}

    def _set_trained_variables(self, variables):
        pass

    def set_model(self, model: torch.nn.Module):
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model
        if self.feature_extractor:
            self.feature_extractor.set_model(model)

    def compute_layerwise_scores(self, inputs) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        Args:
            inputs: a batch of input data to the model

        Returns:
            A dictionary with anomaly scores, each element has shape (batch_size, ).
        """
        if self.feature_extractor:
            features = self.feature_extractor(inputs)
        else:
            features = None
        return self._compute_layerwise_scores(inputs=inputs, features=features)

    def compute_scores(self, inputs) -> torch.Tensor:
        """Compute anomaly scores for the given inputs.

        Args:
            inputs: a batch of input data to the model

        Returns:
            Anomaly scores for the given inputs, of shape (batch_size, )
        """
        scores = self.compute_layerwise_scores(inputs)
        return self._aggregate_scores(scores)

    def _aggregate_scores(
        self, layerwise_scores: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        scores = layerwise_scores.values()
        assert len(scores) > 0
        # Type checker doesn't take into account that scores is non-empty,
        # so thinks this might be a float.
        if self.layer_aggregation == "mean":
            return sum(scores) / len(scores)  # type: ignore
        elif self.layer_aggregation == "max":
            return torch.amax(torch.stack(list(scores)), dim=0)
        else:
            raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    def _collate_fn(self, batch):
        batch = torch.utils.data.default_collate(batch)
        inputs = utils.inputs_from_batch(batch)
        if self.feature_extractor:
            features = self.feature_extractor(inputs)
        else:
            features = None
        return batch, features

    def train(
        self,
        trusted_data: Dataset | None,
        untrusted_data: Dataset | None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        dataloaders = []
        for data in [trusted_data, untrusted_data]:
            if data is None:
                dataloaders.append(None)
            else:
                dataloaders.append(
                    torch.utils.data.DataLoader(
                        data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=self._collate_fn,
                    )
                )

        return self._train(
            trusted_dataloader=dataloaders[0],
            untrusted_dataloader=dataloaders[1],
            **kwargs,
        )

    def eval(
        self,
        dataset: MixedData,
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
        layerwise: bool = False,
        log_yaxis: bool = True,
        show_worst_mistakes: bool = False,
        sample_format_fn: Callable[[Any], Any] | None = None,
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(dataset, MixedData), type(dataset)

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        metrics = defaultdict(dict)
        if save_path is not None:
            model_name = Path(save_path).parts[-1]
        assert 0 < histogram_percentile <= 100

        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

        scores = defaultdict(list)
        anomaly_labels = []
        agreement = []

        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            for batch in test_loader:
                (inputs, _), (new_labels, new_agreements) = batch
                if layerwise:
                    new_scores = self.compute_layerwise_scores(inputs)
                else:
                    # For some detectors, this is what we'd get anyway when calling
                    # compute_layerwise_scores, but for layerwise detectors, we want to
                    # make sure to analyze only the overall scores unless requested
                    # otherwise.
                    new_scores = {"all": self.compute_scores(inputs)}
                for layer, score in new_scores.items():
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    assert score.shape == new_labels.shape
                    scores[layer].append(score)
                anomaly_labels.append(new_labels)
                agreement.append(new_agreements)
        scores = {layer: np.concatenate(scores[layer]) for layer in scores}
        anomaly_labels = np.concatenate(anomaly_labels)
        agreement = np.concatenate(agreement)

        figs = {}

        for layer in scores:
            auc_roc = sklearn.metrics.roc_auc_score(
                y_true=anomaly_labels,
                y_score=scores[layer],
            )
            ap = sklearn.metrics.average_precision_score(
                y_true=anomaly_labels,
                y_score=scores[layer],
            )
            logger.info(f"AUC_ROC ({layer}): {auc_roc:.4f}")
            logger.info(f"AP ({layer}): {ap:.4f}")
            metrics[layer]["AUC_ROC"] = auc_roc
            metrics[layer]["AP"] = ap

            # Calculate the number of negative examples to filter to catch all positives
            sorted_indices = np.argsort(scores[layer])[::-1]
            sorted_labels = anomaly_labels[sorted_indices]
            cut_point = np.where(sorted_labels == 1)[0][-1] + 1
            num_negatives = np.sum(anomaly_labels[:cut_point]==0)
            logger.info(f"Perfect filter remainder ({layer}): {1 - num_negatives/np.sum(anomaly_labels==0)}")
            metrics[layer]["Perfect_filter_remainder"] = 1 - num_negatives/np.sum(anomaly_labels==0)

            if np.any(agreement.astype(bool)):
                auc_roc_agree = sklearn.metrics.roc_auc_score(
                    y_true=anomaly_labels[agreement.astype(bool)],
                    y_score=scores[layer][agreement.astype(bool)],
                )
                ap_agree = sklearn.metrics.average_precision_score(
                    y_true=anomaly_labels[agreement.astype(bool)],
                    y_score=scores[layer][agreement.astype(bool)],
                )
            else:
                auc_roc_agree = ap_agree = 0.5
            logger.info(f"AUC_ROC_AGREE ({layer}): {auc_roc_agree:.4f}")
            logger.info(f"AP_AGREE ({layer}): {ap_agree:.4f}")
            metrics[layer]["AUC_ROC_AGREE"] = auc_roc_agree
            metrics[layer]["AP_AGREE"] = ap_agree

            if np.any(~agreement.astype(bool)):
                auc_roc_disagree = sklearn.metrics.roc_auc_score(
                    y_true=anomaly_labels[~agreement.astype(bool)],
                    y_score=scores[layer][~agreement.astype(bool)],
                )
                ap_disagree = sklearn.metrics.average_precision_score(
                    y_true=anomaly_labels[~agreement.astype(bool)],
                    y_score=scores[layer][~agreement.astype(bool)],
                )
            else:
                auc_roc_disagree = ap_disagree = 0.5
            logger.info(f"AUC_ROC_DISAGREE ({layer}): {auc_roc_disagree:.4f}")
            logger.info(f"AP_DISAGREE ({layer}): {ap_disagree:.4f}")
            metrics[layer]["AUC_ROC_DISAGREE"] = auc_roc_disagree
            metrics[layer]["AP_DISAGREE"] = ap_disagree

            upper_lim = np.percentile(scores[layer], histogram_percentile).item()
            # Usually there aren't extremely low outliers, so we just use the minimum,
            # otherwise this tends to weirdly cut of the histogram.
            lower_lim = scores[layer].min().item()

            bins = np.linspace(lower_lim, upper_lim, num_bins)

            # Visualizations for anomaly scores
            for j, agree_label in enumerate(["Disagree", "Agree"]):
                fig, ax = plt.subplots()
                for i, name in enumerate(["Normal", "Anomalous"]):
                    class_labels = anomaly_labels[agreement == j]
                    vals = scores[layer][agreement == j][class_labels == i]
                    ax.hist(
                        vals,
                        bins=bins,
                        alpha=0.5,
                        label=f"{name} {agree_label}",
                        log=log_yaxis,
                    )
                ax.legend()
                ax.set_xlabel("Anomaly score")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Anomaly score distribution ({layer})\n{model_name}")
                textstr = f"AUROC: {auc_roc:.1%}\n AP: {ap:.1%}"
                props = dict(boxstyle="round", facecolor="white")
                ax.text(
                    0.98,
                    0.80,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=props,
                )
                figs[(layer, agree_label)] = fig

        if show_worst_mistakes:
            for layer, layer_scores in scores.items():
                # "false positives" etc. isn't quite right because there's no threshold
                false_positives = np.argsort(
                    np.where(anomaly_labels == 0, layer_scores, -np.inf)
                )[-10:]
                false_negatives = np.argsort(
                    np.where(anomaly_labels == 1, layer_scores, np.inf)
                )[:10]

                print("\nNormal but high anomaly score:\n")
                for idx in false_positives:
                    sample, anomaly_label = dataset[idx]
                    assert anomaly_label == 0
                    if sample_format_fn:
                        sample = sample_format_fn(sample)
                    print(f"#{idx} ({layer_scores[idx]}): {sample}")
                print("\n====================================")
                print("Anomalous but low anomaly score:\n")
                for idx in false_negatives:
                    sample, anomaly_label = dataset[idx]
                    assert anomaly_label == 1
                    if sample_format_fn:
                        sample = sample_format_fn(sample)
                    print(f"#{idx} ({layer_scores[idx]}): {sample}")

        if not save_path:
            return metrics, figs

        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        for layer, fig in figs.items():
            fig.savefig(save_path / f"histogram_{layer}_{agree_label}.pdf")

        return metrics, figs

    def save_weights(self, path: str | Path):
        logger.info(f"Saving detector to {path}")
        utils.save(self._get_trained_variables(), path)

    def load_weights(self, path: str | Path):
        logger.info(f"Loading detector from {path}")
        self._set_trained_variables(utils.load(path))


class AccuracyDetector(AnomalyDetector):
    """
    Convenience class for evaluating answer accuracy instead of anomaly detection
    """

    def train(self, **kwargs):
        pass
    
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        raise NotImplementedError('This detector calculates scores directly')
    
    def scores(self, batch) -> torch.Tensor:
        inputs = utils.inputs_from_batch(batch)
        encoding = self.model.tokenize(inputs)
        mask = encoding['attention_mask']

        no_token = self.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
        yes_token = self.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
        effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device=self.model.device)

        logits = self.model(inputs).logits[..., effect_tokens][range(len(inputs)), mask.sum(dim=1) - 1]
        return torch.nn.functional.softmax(logits, dim=1)[:, 1]

class IterativeAnomalyDetector(AnomalyDetector):
    def train(
        self,
        trusted_data: Dataset | None,
        untrusted_data: Dataset | None,
        save_path: Path | str | None,
        **kwargs,
    ):
        pass

    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        raise NotImplementedError('This detector calculates scores directly')
    
    def scores(self, batch) -> torch.Tensor:
        inputs = utils.inputs_from_batch(batch)
        encoding = self.model.tokenize(inputs, padding=True)
        mask = encoding['attention_mask']

        # Get logits for 'No' and 'Yes' tokens
        no_token = self.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
        yes_token = self.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
        effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device=self.model.device)

        # Perform a forward pass to get logits at the last non-padded position
        logits = self.model(inputs).logits[..., effect_tokens][range(len(inputs)), mask.sum(dim=1) - 1]
        # Determine the lower and higher logit tokens
        lower_logit_indices = logits.argmin(dim=1)
        higher_logit_indices = 1 - lower_logit_indices  # Since there are only two tokens, this will select the other token

        # Extract the corresponding logits
        lower_logits = logits.gather(1, lower_logit_indices.unsqueeze(-1)).squeeze(-1)
        higher_logits = logits.gather(1, higher_logit_indices.unsqueeze(-1)).squeeze(-1)

        # Modify the batch prompts with the lower logit token text
        lower_logit_tokens = self.model.tokenizer.convert_ids_to_tokens(effect_tokens[lower_logit_indices])
        modified_prompts = [prompt + f'\n\n One possible answer is "{token}"\n\n Is the statement factually correct?' 
                            for prompt, token in zip(inputs, lower_logit_tokens)]

        # Encode the modified prompts
        modified_encoding = self.model.tokenize(modified_prompts, padding=True)
        modified_mask = modified_encoding['attention_mask']

        # Perform a second forward pass
        modified_outputs = self.model(modified_prompts)
        modified_logits = modified_outputs.logits[..., effect_tokens][range(len(modified_prompts)), modified_mask.sum(dim=1) - 1]

        # Extract the logits for the second pass
        lower_logits_second_pass = modified_logits.gather(1, lower_logit_indices.unsqueeze(-1)).squeeze(-1)
        higher_logits_second_pass = modified_logits.gather(1, higher_logit_indices.unsqueeze(-1)).squeeze(-1)

        # Compute the score
        score = (higher_logits - lower_logits) - (higher_logits_second_pass - lower_logits_second_pass)

        return score