import json
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import pdb

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer import utils
from cupbearer.data import MixedData


class AnomalyDetector(ABC):
    def __init__(self, layer_aggregation: str = "mean"):
        # For storing the original detector variables when finetuning
        self.layer_aggregation = layer_aggregation
        self._original_variables = None
        self.trained = False

    def set_model(self, model: torch.nn.Module):
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model
        self.trained = False

    @abstractmethod
    def train(
        self,
        trusted_data: Dataset | None,
        untrusted_data: Dataset | None,
        save_path: Path | str | None,
        **kwargs,
    ):
        """Train the anomaly detector with the given datasets on the given model.

        At least one of trusted_data or untrusted_data must be provided.
        """

    @contextmanager
    def finetune(self, **kwargs):
        """Tune the anomaly detector.

        The finetuned parameters will be stored in this detector alongside the original
        ones. Within the context manager block, the detector will use the finetuned
        parameters (e.g. when calling `eval`). At the end of the block, the finetuned
        parameters will be removed. To store the finetuned parameters permanently,
        you can access the value the context manager yields.

        Might not be available for all anomaly detectors.

        Example:
        ```
        with detector.finetune(normal_dataset, new_dataset) as finetuned_params:
            detector.eval(normal_dataset, new_dataset) # uses finetuned params
            scores = detector.scores(some_other_dataset)
            utils.save(finetuned_params, "finetuned_params")

        detector.eval(normal_dataset, new_dataset) # uses original params
        ```
        """
        self._original_vars = self._get_trained_variables()
        finetuned_vars = self._finetune(**kwargs)
        self._set_trained_variables(finetuned_vars)
        yield finetuned_vars
        if self._original_vars:
            # original_vars might be empty if the detector was never trained
            self._set_trained_variables(self._original_vars)
        self._original_vars = None

    def _finetune(self, **kwargs) -> dict:
        """Finetune the anomaly detector to try to flag the new data as anomalous.

        Should return variables for the detector that can be passed to
        `_set_trained_variables`.
        """
        raise NotImplementedError(
            f"Finetuning not implemented for {self.__class__.__name__}."
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
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(dataset, MixedData), type(dataset)

        dataset.return_anomaly_agreement = True

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # For some methods, such as adversarial abstractions, it might matter how
            # normal/anomalous data is distributed into batches. In that case, we want
            # to mix them by default.
            shuffle=True,
        )

        metrics = defaultdict(dict)
        if save_path is not None:
            model_name = Path(save_path).parts[-1]
        assert 0 < histogram_percentile <= 100

        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

        scores = defaultdict(list)
        labels = defaultdict(list)
        agreement = defaultdict(list)
        answers = defaultdict(list)

        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            for batch in test_loader:
                inputs, (new_labels, new_agreements) = batch
                if layerwise:
                    new_scores = self.layerwise_scores(inputs)
                else:
                    new_scores = {"all": self.scores(inputs)}
                for layer, score in new_scores.items():
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    assert score.shape == new_labels.shape
                    scores[layer].append(score)
                    labels[layer].append(new_labels)
                    agreement[layer].append(new_agreements)
        scores = {layer: np.concatenate(scores[layer]) for layer in scores}
        labels = {layer: np.concatenate(labels[layer]) for layer in labels}
        agreement = {layer: np.concatenate(agreement[layer]) for layer in agreement}
        answers = {layer: np.concatenate(answers[layer]) for layer in answers}

        figs = {}

        for layer in scores:
            auc_roc = sklearn.metrics.roc_auc_score(
                y_true=labels[layer],
                y_score=scores[layer],
            )
            ap = sklearn.metrics.average_precision_score(
                y_true=labels[layer],
                y_score=scores[layer],
            )
            logger.info(f"AUC_ROC ({layer}): {auc_roc:.4f}")
            logger.info(f"AP ({layer}): {ap:.4f}")
            metrics[layer]["AUC_ROC"] = auc_roc
            metrics[layer]["AP"] = ap

            # Calculate the number of negative examples to filter to catch all positives
            sorted_indices = np.argsort(scores[layer])[::-1]
            sorted_labels = labels[layer][sorted_indices]
            cut_point = np.where(sorted_labels == 1)[0][-1] + 1
            num_negatives = np.sum(labels[layer][:cut_point]==0)
            logger.info(f"Perfect filter remainder ({layer}): {1 - num_negatives/np.sum(labels[layer]==0)}")
            metrics[layer]["Perfect_filter_remainder"] = 1 - num_negatives/np.sum(labels[layer]==0)

            if np.any(agreement[layer].astype(bool)):
                auc_roc_agree = sklearn.metrics.roc_auc_score(
                    y_true=labels[layer][agreement[layer].astype(bool)],
                    y_score=scores[layer][agreement[layer].astype(bool)],
                )
                ap_agree = sklearn.metrics.average_precision_score(
                    y_true=labels[layer][agreement[layer].astype(bool)],
                    y_score=scores[layer][agreement[layer].astype(bool)],
                )
            else:
                auc_roc_agree = ap_agree = 0.5
            logger.info(f"AUC_ROC_AGREE ({layer}): {auc_roc_agree:.4f}")
            logger.info(f"AP_AGREE ({layer}): {ap_agree:.4f}")
            metrics[layer]["AUC_ROC_AGREE"] = auc_roc_agree
            metrics[layer]["AP_AGREE"] = ap_agree

            if np.any(~agreement[layer].astype(bool)):
                auc_roc_disagree = sklearn.metrics.roc_auc_score(
                    y_true=labels[layer][~agreement[layer].astype(bool)],
                    y_score=scores[layer][~agreement[layer].astype(bool)],
                )
                ap_disagree = sklearn.metrics.average_precision_score(
                    y_true=labels[layer][~agreement[layer].astype(bool)],
                    y_score=scores[layer][~agreement[layer].astype(bool)],
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
                    class_labels = labels[layer][agreement[layer] == j]
                    vals = scores[layer][agreement[layer] == j][class_labels == i]
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

            if not save_path:
                return metrics, figs

        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Everything from here is just saving metrics and creating figures
        # (which we skip if they aren't going to be saved anyway).
        with open(save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        for layer, fig in figs.items():
            fig.savefig(save_path / f"histogram_{layer}_{agree_label}.pdf")

        return metrics, figs

    @abstractmethod
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        You can just raise a NotImplementedError here for detectors that don't compute
        layerwise scores. In that case, you need to override `scores`. For detectors
        that can compute layerwise scores, you should override this method instead
        of `scores` since allows some additional metrics to be computed.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            A dictionary with anomaly scores, each element has shape (batch, ).
        """

    def scores(self, batch) -> torch.Tensor:
        """Compute anomaly scores for the given inputs.

        If you override this, then your implementation of `layerwise_scores()`
        needs to raise a NotImplementedError. Implementing both this and
        `layerwise_scores()` is not supported.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            A batch of anomaly scores for the inputs.
        """
        scores = self.layerwise_scores(batch).values()
        assert len(scores) > 0
        # Type checker doesn't take into account that scores is non-empty,
        # so thinks this might be a float.
        if self.layer_aggregation == "mean":
            return sum(scores) / len(scores)  # type: ignore
        elif self.layer_aggregation == "max":
            return torch.amax(torch.stack(list(scores)), dim=0)
        elif self.layer_aggregation == "sum":
            return torch.sum(torch.stack(list(scores)), dim=0)
        else:
            raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    def _get_trained_variables(self, saving: bool = False):
        return {}

    def _set_trained_variables(self, variables):
        pass

    def save_weights(self, path: str | Path):
        logger.info(f"Saving detector to {path}")
        utils.save(self._get_trained_variables(saving=True), path)

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