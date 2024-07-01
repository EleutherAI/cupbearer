from typing import Callable, Any
from tqdm import tqdm

import torch

from cupbearer import utils
from cupbearer.utils.classifier import Classifier
from cupbearer.detectors.activation_based import ActivationBasedDetector, ActivationCache
from cupbearer.detectors.statistical.helpers import mahalanobis_from_data

# Thanks GPT-4o
# Procedure:
"""
> I'm probing the activations of a language model and I want to distinguish misconceptions from true answers. 
I want to create a true/false question such that "question question question? yes" is likely to be continued by misconceived/unreliable text.
The misconception should be one that I might actually encounter if I went out and interviewed people. Generate 10 such misconceptions.
Remember to make them diverse, but commonly believed.
...
> Rephrase them to make the phrasing more diverse and gramatically correct
...
I manually inverted them all because GPT-4 frequently rephrased them without changing the answer (and checked that the answers were correct)
...
> {Previous misconceptions pasted into new chat session} Now, suggest 10 diverse examples of error. These should not be scientific misconceptions, but other kinds of error e.g. mathematical or logical; they should be somewhat common kinds of error.
...
> Make the phrasing more diverse
...
Manually inverted again + checked answers.
"""

misconception_pairs = [
    ("Do vaccines cause autism? Yes", "Do vaccines cause autism? No"),
    ("Do we only use 10% of our brains? Yes", "Do we only use 10% of our brains? No"),
    ("Does lightning never strike the same place twice? Yes", "Does lightning never strike the same place twice? No"),
    ("Does cracking your knuckles cause arthritis? Yes", "Does cracking your knuckles cause arthritis? No"),
    ("Do goldfish have a memory span of only 3 seconds? Yes", "Do goldfish have a memory span of only 3 seconds? No"),
    ("Can the Great Wall of China be seen from space? Yes", "Can the Great Wall of China be seen from space? No"),
    ("Does shaving hair make it grow back thicker? Yes", "Does shaving hair make it grow back thicker? No"),
    ("Is Mount Everest the closest point to space? Yes", "Is Mount Everest the closest point to space? No"),
    ("Do bulls get angry when they see red? Yes", "Do bulls get angry when they see red? No"),
    ("Vaccines don't cause autism, right? No", "Vaccines don't cause autism, right? Yes"),
    ("It's false that we use only 10% of our brains, correct? No", "It's false that we use only 10% of our brains, correct? Yes"),
    ("Lightning can strike the same place twice, right? No", "Lightning can strike the same place twice, right? Yes"),
    ("Cracking your knuckles doesn't cause arthritis, right? No", "Cracking your knuckles doesn't cause arthritis, right? Yes"),
    ("Goldfish have a memory span of more than 3 seconds, right? No", "Goldfish have a memory span of more than 3 seconds, right? Yes"),
    ("The Great Wall of China is not visible from space, right? No", "The Great Wall of China is not visible from space, right? Yes"),
    ("Shaving hair doesn't make it grow back thicker, right? No", "Shaving hair doesn't make it grow back thicker, right? Yes"),
    ("Mount Everest is not the closest point to space, right? No", "Mount Everest is not the closest point to space, right? Yes"),
    ("Bulls don't necessarily get angry when they see red, right? No", "Bulls don't necessarily get angry when they see red, right? Yes"),
    ("Does 2+2 equal 5? Yes", "Does 2+2 equal 5? No"),
    ("Is Berlin the capital of France? Yes", "Is Berlin the capital of France? No"),
    ("Is H2O composed of two oxygen atoms and one hydrogen atom? Yes", "Is H2O composed of two oxygen atoms and one hydrogen atom? No"),
    ("Is the Earth a flat surface? Yes", "Is the Earth a flat surface? No"),
    ("Does 10 divided by 2 give you 3? Yes", "Does 10 divided by 2 give you 3? No"),
    ("Is the square root of 16 actually 5? Yes", "Is the square root of 16 actually 5? No"),
    ("Do the angles in a triangle add up to 200 degrees? Yes", "Do the angles in a triangle add up to 200 degrees? No"),
    ("Is the moon really made of cheese? Yes", "Is the moon really made of cheese? No"),
    ("Is 7 not a prime number? Yes", "Is 7 not a prime number? No"),
    ("Does water boil at 50 degrees Celsius? Yes", "Does water boil at 50 degrees Celsius? No"),
    ("Does 2+2 equal 4? No", "Does 2+2 equal 4? Yes"),
    ("Is Berlin not the capital of France? No", "Is Berlin not the capital of France? Yes"),
    ("Is water not made up of two oxygen atoms and one hydrogen atom? No", "Is water not made up of two oxygen atoms and one hydrogen atom? Yes"),
    ("Is the Earth not flat? No", "Is the Earth not flat? Yes"),
    ("Does 10 divided by 2 give you 5? No", "Does 10 divided by 2 give you 5? Yes"),
    ("Is the square root of 16 equal to 4? No", "Is the square root of 16 equal to 4? Yes"),
    ("Is the sum of angles in a triangle less than 200 degrees? No", "Is the sum of angles in a triangle less than 200 degrees? Yes"),
    ("Is the moon not made of cheese? No", "Is the moon not made of cheese? Yes"),
    ("Is 7 a prime number? No", "Is 7 a prime number? Yes"),
    ("Does water boil at more than 50 degrees Celsius? No", "Does water boil at more than 50 degrees Celsius? Yes"),
]   

class MisconceptionContrastDetector(ActivationBasedDetector):
    def train(self, trusted_data=None, untrusted_data=None, save_path=None, **kwargs):
        misconceptions, correct_conceptions = zip(*misconception_pairs)
        misconceptions_activations = self.get_activations(misconceptions)
        correct_conceptions_activations = self.get_activations(correct_conceptions)
        self.classifier = {}
        self.input_dim = {}

        for layer in self.activation_names:
            x = torch.cat([misconceptions_activations[layer], correct_conceptions_activations[layer]], dim=0)
            y = torch.cat([torch.ones(len(misconceptions)), torch.zeros(len(correct_conceptions))], dim=0)
            input_dim = x.shape[-1]
            self.input_dim[layer] = input_dim
            self.classifier[layer] = Classifier(input_dim=input_dim, device=self.model.device)
            self.classifier[layer].fit_cv(x.to(self.model.device), y.to(self.model.device))

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        scores = {}

        for layer in self.activation_names:
            probabilities = self.classifier[layer].to(activations[layer].device).forward(activations[layer])
            scores[layer] = probabilities

        return scores

    def _get_trained_variables(self, saving: bool = False):
        return {
            "input_dim": self.input_dim,
            "classifier_state_dict": {layer: self.classifier[layer].state_dict() for layer in self.activation_names},
        }

    def _set_trained_variables(self, variables):
        if variables["classifier_state_dict"]:
            input_dim = variables["input_dim"]
            self.classifier = {}
            for layer in self.activation_names:
                self.classifier[layer] = Classifier(input_dim=input_dim[layer])
                self.classifier[layer].load_state_dict(variables["classifier_state_dict"][layer])


class ProbeTrajectoryDetector(ActivationBasedDetector):
    def train(self, trusted_data=None, untrusted_data=None, save_path=None, batch_size=16, **kwargs):
        dl = torch.utils.data.DataLoader(trusted_data, batch_size=batch_size, shuffle=True)

        activations = {layer: [] for layer in self.activation_names}
        answers = []

        for batch in tqdm(dl):
            inputs, labels = batch
            new_activations = self.get_activations(batch)
            for layer in self.activation_names:
                activations[layer].append(new_activations[layer].cpu())
            answers.append(labels)

        activations = {layer: torch.cat(activations[layer], dim=0) for layer in self.activation_names}
        answers = torch.cat(answers, dim=0)

        self.classifier = {}
        self.input_dim = {}
        self.trajectory = {}

        for layer in tqdm(self.activation_names):
            input_dim = activations[layer].shape[-1]
            self.input_dim[layer] = input_dim
            self.classifier[layer] = Classifier(input_dim=input_dim, device=self.model.device)
            self.classifier[layer].fit_cv(activations[layer].to(self.model.device), answers.to(self.model.device))
            with torch.no_grad():
                self.trajectory[layer] = self.classifier[layer].forward(activations[layer].to(self.model.device))

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        test_trajectory = {}

        with torch.no_grad():
            for layer in self.activation_names:
                probabilities = self.classifier[layer].to(self.model.device).forward(activations[layer].to(self.model.device))
                test_trajectory[layer] = probabilities

        train_trajectories = torch.cat([self.trajectory[layer].unsqueeze(-1) for layer in self.activation_names 
                                        if (torch.isfinite(self.trajectory[layer]).all() and torch.isfinite(test_trajectory[layer]).all())], dim=-1)
        test_trajectory = torch.cat([test_trajectory[layer].unsqueeze(-1) for layer in self.activation_names 
                                     if (torch.isfinite(self.trajectory[layer]).all() and torch.isfinite(test_trajectory[layer]).all())], dim=-1)

        return mahalanobis_from_data({'all': test_trajectory}, {'all': train_trajectories}, relative=False)

    def _get_trained_variables(self, saving: bool = False):
        return {
            "input_dim": self.input_dim,
            "classifier_state_dict": {layer: self.classifier[layer].state_dict() for layer in self.activation_names},
            "trajectory": self.trajectory,
        }

    def _set_trained_variables(self, variables):
        if variables["classifier_state_dict"]:
            input_dim = variables["input_dim"]
            self.classifier = {}
            for layer in self.activation_names:
                self.classifier[layer] = Classifier(input_dim=input_dim[layer])
                self.classifier[layer].load_state_dict(variables["classifier_state_dict"][layer])
        self.trajectory = variables["trajectory"]