# `cupbearer` 🍷
`cupbearer` is a Python library for
[mechanistic anomaly detection](https://www.alignmentforum.org/posts/vwt3wKXWaCvqZyF74/mechanistic-anomaly-detection-and-elk).
Its main purpose is to make it easy to implement either a new mechanistic anomaly
detection *task*, or a new detection *method*, and evaluate it against a suite of existing
tasks or methods. To that end, the library provides:
- A clearly defined interface for tasks and detectors to ensure compatibility
- Scripts and other helpers for training and evaluating detectors that satisfy this interface
  (to reduce the amount of boilerplate code you need to write)
- Implementations of several tasks and detectors as baselines (currently not that many)

Contributions of new tasks or detectors are very welcome!
See the [developer guide](docs/getting_started.md) to get started.

## Installation
The easy way: inside a virtual environment with Python >= 3.10, run
```bash
pip install git+https://github.com/ejnnr/cupbearer.git
```
(You could also `pip install cupbearer`, but note that the library is under heavy
development and the PyPi version will often be outdated.)

Alternatively, if you're going to do development work on the library itself,
see [the developer guide](docs/getting_started.md).

### Notes on Pytorch
Depending on what platform you're on, you may need to install Pytorch separately *before*
installing `cupbearer`, in particular if you want to control CUDA version etc.

## Running experiments
We provide scripts in `cupbearer.scripts` for more easily running experiments.
See [the demo notebook](notebooks/simple_demo.ipynb) for a quick example of how to use them---this is likely
also the best way to get an overview of how the components of `cupbearer` fit together.

These "scripts" are Python functions and designed to be used from within Python,
e.g. in a Jupyter notebook or via [submitit](https://github.com/facebookincubator/submitit/tree/main)
if on Slurm. But of course you could also write a simple Python wrapper and then use
them from the CLI. The scripts are designed to be pretty general,
which sometimes comes at the cost of being a bit verbose---we recommend writing helper
functions for your specific use case on top of the general script interface.
Of course you can also use the components of `cupbearer` directly without going through
any of the scripts.

## Whence the name?
Just like a cupbearer tastes wine to avoid poisoning the king, mechanistic anomaly
detection methods taste new inputs to check whether they're poisoned. (Or more generally,
anomalous in terms of how they affect the king ... I mean, model. I admit the analogy
is becoming a bit strained here.)

## Quirky language models

The quirky language models task uses [quirky datasets](https://arxiv.org/abs/2312.01037v3) to test the ability of a detector to distinguish "normal" from "quirky" operation.

Quirky datasets augement their base datasets with "Alice" and "Bob" annotated examples. "Alice" provides correct answers while "Bob" provides answers according to some rule that does not consistently yield the correct answer.

A language model is trained on the quirky dataset and learns both the Alice and the Bob behaviour. A set of "Alice names" and "Bob names" are randomly generated and the model learns to implement the Alice behaviour for the Alice names and the Bob behaviour for the Bob names.

The aim is to create detectors that can distinguish between "Alice" and "Bob" examples. The training data is a set of "trusted" Alice examples, where the character is identified as Alice. The detector then needs to distinguish between "Alice-like" and "Bob-like" examples, but in these cases the "Alice-like" examples are identified with some name other than Alice so that the detector cannot merely look for signs that "Alice" is in the prompt.

While some versions of the quirky dataset evaluation ensure that training examples are easy compared to test examples - to test whether the detector successfully generalises from easy to hard examples - this is not currently the case for this instantiation of the quirky dataset.