from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Callable
from einops import rearrange
from cupbearer import utils
import pdb
from pathlib import Path
import gc
from abc import ABC, abstractmethod

from cupbearer.detectors.statistical.helpers import local_outlier_factor, mahalanobis_from_data
from cupbearer.detectors.statistical.statistical import StatisticalDetector
from cupbearer.detectors.activation_based import CacheBuilder

class TrajectoryDetector(StatisticalDetector, ABC):
    """Generic abstract detector that records prediction trajectories from a tuned lens."""
    @abstractmethod
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        pass

    def __init__(
            self, 
            layers: list[int], 
            lens_dir: str = Path('/mnt/ssd-1/nora/tuned-lens/mistral'),
            base_model_name: str = "mistralai/Mistral-7B-v0.1",
            seq_len: int = 15,
            cache: CacheBuilder = None
            ):
        super().__init__([], cache=cache)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.lens = TunedLens.from_model_and_pretrained(base_model, lens_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # tokens_for_vocab = ["Yes", "No", "yes", "no", "false", "true", "False", "True"]

        # self.vocab = torch.unique(torch.tensor(self.tokenizer.encode(' '.join(tokens_for_vocab) + '.' + '.'.join(tokens_for_vocab))))

        tokens_for_vocab = ["Yes", "No"]
        self.vocab = torch.unique(torch.tensor(self.tokenizer.encode(' '.join(tokens_for_vocab))))[1:]
        self.vocab_size = len(self.vocab)

        del base_model
        gc.collect()

        self.seq_len = seq_len
        self.layers = layers

        self._trajectories = {
            k: torch.empty((len(self.layers), self.seq_len * self.tokenizer.vocab_size))
            for k in self.layers
        }

    def init_variables(self, trajectory_sizes: dict[str, torch.Size], device):
        pass

    def batch_update(self, trajectories: dict[str, torch.Tensor]):
        for k, trajectory in trajectories.items():
            assert trajectory.ndim == 2, trajectory.shape
            self._trajectories[k] = torch.cat(
                (self._trajectories[k], trajectory), dim=0
            )

    def train(self, trusted_data, untrusted_data, **kwargs):
        super().train(
            trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process

        with torch.inference_mode():
            self.trajectories = self._trajectories

    def get_activations(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)
        selected_tokens = slice(-self.seq_len, None)
        prediction_trajectories = torch.zeros((len(inputs), self.lens.config.num_hidden_layers + 1, self.seq_len * self.tokenizer.vocab_size))

        for i, inp_id in enumerate(inputs):
            tokenized_input = self.tokenizer.encode(inp_id)
            prediction_trajectory = torch.Tensor(PredictionTrajectory.from_lens_and_model(
                self.lens.to(self.model.hf_model.device),
                self.model.hf_model,
                tokenizer=self.tokenizer,
                input_ids=tokenized_input,
            ).slice_sequence(selected_tokens).log_probs).reshape(self.lens.config.num_hidden_layers + 1, self.seq_len * self.tokenizer.vocab_size)

            assert torch.isnan(prediction_trajectory).any() == False

            prediction_trajectories[i] = prediction_trajectory

        trajectory_dict = {k: prediction_trajectories[:, k] for k in self.layers}

        return trajectory_dict

    def _get_trained_variables(self, saving: bool = False):
        return{
            "trajectories": self.trajectories,
        }

    def _set_trained_variables(self, variables):
        self.trajectories = variables["trajectories"]


class LOFTrajectoryDetector(TrajectoryDetector):
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        test_trajectories = self.get_activations(batch)
        batch_size = next(iter(test_trajectories.values())).shape[0]

        # Select just the tokens on interest in the vocab
        for k, test_trajectory in test_trajectories.items():
            test_trajectories[k] = test_trajectory.reshape(batch_size, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(batch_size, self.seq_len * self.vocab_size)

            assert torch.isnan(test_trajectory).any() == False

        distances = local_outlier_factor(
            test_trajectories,
            {k: v.reshape(-1, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(-1, self.seq_len * self.vocab_size)
             for k, v in self.trajectories.items()},
        )

        for k, v in distances.items():
            # Unflatten distances so we can take the mean over the independent axis
            distances[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=batch_size
            ).mean(dim=1)

        return distances

class MahaTrajectoryDetector(TrajectoryDetector):
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        test_trajectories = self.get_activations(batch)
        batch_size = next(iter(test_trajectories.values())).shape[0]

        # Select just the tokens on interest in the vocab
        for k, test_trajectory in test_trajectories.items():
            test_trajectories[k] = test_trajectory.reshape(batch_size, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(batch_size, self.seq_len * self.vocab_size)

            assert torch.isnan(test_trajectory).any() == False
        learned_trajectories = {k: v.reshape(-1, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(-1, self.seq_len * self.vocab_size)
             for k, v in self.trajectories.items()}
        
        distances = mahalanobis_from_data(test_trajectories, learned_trajectories)

        for k, v in distances.items():
            # Unflatten distances so we can take the mean over the independent axis
            distances[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=batch_size
            ).mean(dim=1)

        return distances