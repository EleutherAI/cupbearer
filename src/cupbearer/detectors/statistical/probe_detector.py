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
            cache: CacheBuilder = None
            ):
        # Hardcoded for now, we want to select multiple eventually
        self.layers = [29]

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.lens = TunedLens.from_model_and_pretrained(base_model, lens_dir)
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
        **kwargs,
    ):
        assert trusted_data is not None
        self.model.hf_model.config.output_hidden_states = True
        self.lens.to(self.model.hf_model.device)
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
            noise_batch_size = self.noise[list(self.noise.keys())[0]][0].shape[0]
        else:
            noise_batch_size = self.noise[list(self.noise.keys())[0]].shape[0]
        
        self._effects = {
            name: torch.zeros(len(trusted_data), noise_batch_size * 32, device=device)
            for name, shape in self.shapes.items()
        }
        self._effects['out'] = torch.zeros(len(trusted_data), 1, device=device)

        dataloader = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size)
        torch.cuda.empty_cache()

        for i, batch in tqdm(enumerate(dataloader)):
            inputs = utils.inputs_from_batch(batch)
            with atp(self.model, self.noise, head_dim=128) as effects:
                outputs = self.model(inputs)
                logits_model = outputs.logits[:, -1, self.vocab].diff(1)
                logits_model.sum().backward()
            
            self.model.zero_grad()

            del outputs
            torch.cuda.empty_cache()

            with atp(self.model, self.noise, head_dim=128) as probe_effects:
                outputs = self.model(inputs)
                hidden_states = outputs.hidden_states[self.layers[0]]
                logits_lens = self.lens.forward(hidden_states, self.layers[0])[:, -1, self.vocab].diff(1)
                logits_lens.sum().backward()
                diff = logits_model.detach() - logits_lens.detach()
                self._effects['out'][i: i + len(batch[0])] = diff.cpu()
            
            self.model.zero_grad()
            self.lens.zero_grad()
            
            del outputs, hidden_states, logits_lens, logits_model
            torch.cuda.empty_cache()

            for name, probe_effect in probe_effects.items():
                # Get the effect at the last token
                effect = effects[name][:, :, -1].detach()
                # Merge the last dimensions
                effect = effect.reshape(batch_size, -1)
                probe_effect = probe_effect[:, :, -1].detach().reshape(batch_size, -1)
                self._effects[name][i: i + len(batch[0])] = (effect - probe_effect).cpu()
            del effects, probe_effects
            torch.cuda.empty_cache()

        self.post_train()

    def post_train(self):
        self.effects = self._effects

    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)
        test_features = defaultdict(lambda: torch.empty((len(batch), 1)))
        self.model.hf_model.config.output_hidden_states = True
        self.lens.to(self.model.hf_model.device)
        batch_size = len(batch[0])

        with torch.enable_grad():
            with atp(self.model, self.noise, head_dim=128) as effects:
                outputs = self.model(inputs)
                logits_model = outputs.logits[:, -1, self.vocab].diff(1)
                hidden_states = outputs.hidden_states[self.layers[0]]
                logits_model.sum().backward()

            with atp(self.model, self.noise, head_dim=128) as probe_effects:
                outputs = self.model(inputs)
                hidden_states = outputs.hidden_states[self.layers[0]]
                logits_lens = self.lens.forward(hidden_states, self.layers[0])[:, -1, self.vocab].diff(1)
                logits_lens.sum().backward()
                diff = logits_model - logits_lens
                test_features['out'] = diff

        for name, probe_effect in probe_effects.items():
            # Get the effect at the last token
            effect = effects[name][:, :, -1].reshape(batch_size, -1)
            probe_effect = probe_effect[:, :, -1].reshape(batch_size, -1)
            effect_diff = effect - probe_effect
            test_features[name] = effect_diff

        scores = {
            k: self._individual_layerwise_score(
                k,
                test_features[k]
            )
            for k in test_features.keys()
        }

        for k, v in scores.items():
            # Unflatten distances so we can take the mean over the independent axis
            scores[k] = rearrange(
                v, "(batch independent) -> batch independent", batch=len(batch[0])
            ).mean(dim=1)

        return scores

    def _get_trained_variables(self, saving: bool = False):
        return{
            "effects": self.effects,
            "noise": self.noise
        }

    def _set_trained_variables(self, variables):
        self.effects = variables["effects"]
        self.noise = variables["noise"]

