from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Callable
from einops import rearrange
from cupbearer import utils
import gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Any
import pdb

from cupbearer.detectors.statistical.trajectory_detector import TrajectoryDetector, mahalanobis_from_data
from cupbearer.detectors.statistical.atp_detector import AttributionDetector
from cupbearer.detectors.statistical.atp_detector import atp
from cupbearer.detectors.activation_based import CacheBuilder
from cupbearer.data import HuggingfaceDataset


def probe_error(test_features, learned_features):
    return test_features.abs().topk(max(1, int(0.01 * test_features.size(1))), dim=1).values.mean(dim=1)

class SimpleProbeDetector(TrajectoryDetector):
    """Detects anomalous examples if the probabilities of '_Yes' and '_No' tokens differ between the middle and the output."""
    def __init__(
            self, 
            lens_dir: str = Path('/mnt/ssd-1/nora/tuned-lens/mistral'),
            base_model_name: str = "mistralai/Mistral-7B-v0.1",
            seq_len: int = 1
            ):
        # Hardcoded for Mistral-7B
        layers = [26, 32]

        super().__init__(layers, lens_dir, base_model_name, seq_len)

        self._trajectories = {
            k: torch.empty((len(self.layers), self.seq_len * self.tokenizer.vocab_size))
            for k in self.layers[:-1]
        }

        # Ensure the vocab is ['_Yes', '_No']   
        tokens_for_vocab = ["Yes", "No"]
        self.vocab = torch.unique(torch.tensor(self.tokenizer.encode(' '.join(tokens_for_vocab))))[1:]
        self.vocab_size = len(self.vocab)

    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        test_trajectories = self.get_activations(batch)
        batch_size = next(iter(test_trajectories.values())).shape[0]

        # Select just the tokens on interest in the vocab
        for k, test_trajectory in test_trajectories.items():
            test_trajectories[k] = test_trajectory.reshape(batch_size, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(batch_size, self.seq_len * self.vocab_size)

            assert torch.isnan(test_trajectory).any() == False

        learned_trajectories = {k: v.reshape(-1, self.seq_len, self.tokenizer.vocab_size).index_select(2, self.vocab).reshape(-1, self.seq_len * self.vocab_size)
             for k, v in self.trajectories.items()}
        learned_trajectories[self.layers[0]] = torch.clamp(
            learned_trajectories[self.layers[1]] - learned_trajectories[self.layers[0]],
            learned_trajectories[self.layers[0]].quantile(0.05),
            learned_trajectories[self.layers[0]].quantile(0.95)
        )
        test_trajectories[self.layers[0]] = torch.clamp(
            test_trajectories[self.layers[1]] - test_trajectories[self.layers[0]],
            learned_trajectories[self.layers[0]].quantile(0.05),
            learned_trajectories[self.layers[0]].quantile(0.95)
        )

        del learned_trajectories[self.layers[1]], test_trajectories[self.layers[1]]

        distances = mahalanobis_from_data(
            test_trajectories,
            learned_trajectories,
        )

        for k, v in distances.items():
            # Unflatten distances so we can take the mean over the independent axis
            distances[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=batch_size
            ).mean(dim=1)

        return distances

class AtPProbeDetector(AttributionDetector):
    """Detects anomalous examples if the probabilities of '_Yes' and '_No' tokens can be caused to differ between the middle and the output."""
    def __init__(
            self,
            shapes: dict[str, int],
            lens_dir: str = Path('/mnt/ssd-1/nora/tuned-lens/mistral'),
            base_model_name: str = "mistralai/Mistral-7B-v0.1",
            seq_len: int = 1,
            activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
            | None = None,
            distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = probe_error,
            ablation: str = 'mean',
            cache: CacheBuilder = None,
            ):
        # Hardcoded for now, we want to select multiple eventually
        self.predictive_layers = (['hf_model.model.layers.2.input_layernorm.input'] + 
                                  [f'hf_model.model.layers.{layer}.input_layernorm.input' for layer in range(6,32,5)])

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.distance_function = distance_function
        del base_model
        gc.collect()

        super().__init__(shapes, lambda x: x, effect_capture_method='atp', effect_capture_args={'ablation': ablation, 'n_pcs': 10}, activation_processing_func=activation_processing_func, cache=cache)

        # Ensure the vocab is ['_Yes', '_No']   
        tokens_for_vocab = ["Yes", "No"]
        self.vocab = torch.unique(torch.tensor(self.tokenizer.encode(' '.join(tokens_for_vocab))))[1:]
        self.vocab_size = len(self.vocab)

    def _individual_layerwise_score(self, name: str, effects: torch.Tensor):
        return self.distance_function(effects, self.effects[name].to(effects.device))

    @torch.enable_grad()
    def train(
        self,
        trusted_data: torch.utils.data.Dataset,
        untrusted_data: torch.utils.data.Dataset | None,
        save_path: Path | str | None,
        batch_size: int = 1,
        subset_size: int = 3000,
        **kwargs,
    ):
        assert trusted_data is not None
        self.model.hf_model.config.output_hidden_states = True
        dtype = self.model.hf_model.dtype
        device = self.model.hf_model.device

        # Why shape[-2]? We are going to sum over the last dimension during attribution
        # patching. We'll then use the second-to-last dimension as our main dimension
        # to fit Gaussians to (all earlier dimensions will be summed out first).
        # This is kind of arbitrary and we're putting the onus on the user to make
        # sure this makes sense.

        with torch.no_grad():
            self.noise = self.get_noise_tensor(trusted_data, batch_size, device, dtype)

        if self.effect_capture_args['ablation'] == 'pcs':
            noise_batch_size = next(iter(self.noise.values()))[0].shape[0]
        else:
            noise_batch_size = next(iter(self.noise.values())).shape[0]
        
        def layer_num(layer_name):
            return int(layer_name.split('.')[3])

        activation_layer_nums = [layer_num(name) for name in self.activation_names]
        predictive_layer_nums = [layer_num(layer) for layer in self.predictive_layers]

        self._effects = {
            name: torch.zeros(
                len(trusted_data), 
                noise_batch_size * 32 * sum(1 for a_num in activation_layer_nums if a_num < p_num),
                device=device
            )
            for name, p_num in zip(self.predictive_layers, predictive_layer_nums)
        }

        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size)

        indices = torch.randperm(len(trusted_data))[:subset_size]
        subset = HuggingfaceDataset(
            trusted_data.hf_dataset.select(indices),
            text_key=trusted_data.text_key,
            label_key=trusted_data.label_key
        )
        lens_dl = torch.utils.data.DataLoader(subset, batch_size=batch_size)

        # Get activations from the predictive layers
        self.intervention_layers = self.activation_names
        self.activation_names = self.predictive_layers

        activations = {layer: [] for layer in self.predictive_layers}
        answers = []

        for batch in tqdm(lens_dl):
            inputs, labels = batch
            new_activations = self.get_activations(batch)
            for layer in self.activation_names:
                activations[layer].append(new_activations[layer].cpu())
            answers.append(labels)

        activations = {layer: torch.cat(activations[layer], dim=0) for layer in self.activation_names}
        answers = torch.cat(answers, dim=0)

        self.lens = {}

        for layer in self.predictive_layers:
            self.lens[layer] = utils.classifier.Classifier(input_dim=activations[layer].shape[-1], device=self.model.device)
            self.lens[layer].fit_cv(activations[layer].to(self.model.device), answers.to(self.model.device))

        del activations, answers
        gc.collect()
        torch.cuda.empty_cache()

        for i, batch in tqdm(enumerate(dataloader)):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=128) as effects:
                outputs = self.model(inputs)
                logits_model = outputs.logits[:, -1, self.vocab].diff(1)
                logits_model.sum().backward()

            effects = torch.cat(list(effects.values()), dim=-1)[:, :, -1].reshape(batch_size, -1)

            probe_effect_dict = {}

            # This is pretty inefficient -- a forward and backward pass for each probe layer
            for layer in self.predictive_layers:
                with atp(self.model, self.noise, head_dim=128) as probe_effects:
                    activations = self._get_activations_no_cache(inputs, no_grad=False)
                    logits_lens = self.lens[layer].forward(activations[layer])
                    logits_lens.sum().backward()
                probe_effect_dict[layer] = torch.cat(list(probe_effects.values()), dim=-1)[:, :, -1].reshape(batch_size, -1)

            for name, probe_effect in probe_effect_dict.items():
                self._effects[name][i: i + len(batch[0])] = (effects[:, :probe_effect.shape[1]] - probe_effect).cpu()

        self.post_train()

    def post_train(self):
        self.effects = self._effects

    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)
        test_features = defaultdict(lambda: torch.empty((len(batch[0]), 1)))
        self.model.hf_model.config.output_hidden_states = True
        for lens in self.lens.values():
            lens.to(self.model.hf_model.device)
        batch_size = len(batch[0])

        self.intervention_layers = self.activation_names
        self.activation_names = self.predictive_layers

        with torch.enable_grad():
            with atp(self.model, self.noise, head_dim=128) as effects:
                outputs = self.model(inputs)
                logits_model = outputs.logits[:, -1, self.vocab].diff(1)
                logits_model.sum().backward()

            effects = torch.cat(list(effects.values()), dim=-1)[:, :, -1].reshape(batch_size, -1)

            probe_effect_dict = {}
            for layer in self.predictive_layers:
                with atp(self.model, self.noise, head_dim=128) as probe_effects:
                    activations = self._get_activations_no_cache(inputs, no_grad=False)
                    logits_lens = self.lens[layer].forward(activations[layer])
                    logits_lens.sum().backward()
                probe_effect_dict[layer] = torch.cat(list(probe_effects.values()), dim=-1)[:, :, -1].reshape(batch_size, -1)

            for name, probe_effect in probe_effect_dict.items():
                test_features[name] = (effects[:, :probe_effect.shape[1]] - probe_effect).cpu()

        scores = {
            k: self._individual_layerwise_score(k, v)
            for k, v in test_features.items()
        }

        for k, v in scores.items():
            scores[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=batch_size
            ).mean(dim=1)

        return scores

    def _get_trained_variables(self, saving: bool = False):
        return{
            "effects": self.effects,
            "noise": self.noise,
            "lens": self.lens
        }

    def _set_trained_variables(self, variables):
        self.effects = variables["effects"]
        self.noise = variables["noise"]
        self.lens = variables["lens"]
