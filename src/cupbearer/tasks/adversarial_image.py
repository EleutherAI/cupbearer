from loguru import logger
from torchvision import datasets
import torch
from autoattack import AutoAttack

from .task import Task

def adversarial_image_task(
    dataset: str = "cifar10",
    model_name: str = "Standard",
    threat_model: str = "Linf",
    epsilon: float = 8/255,
    n_train_examples: int = 500,
    n_test_examples: int = 50,
    device: str = "cuda",
):
    from robustbench.data import _load_dataset, PREPROCESSINGS
    from robustbench.utils import load_model

    ########################
    # Load model and data
    ########################

    if dataset.lower() != "cifar10":
        raise NotImplementedError("Only CIFAR10 is currently supported")

    model = load_model(model_name=model_name, dataset=dataset, threat_model=threat_model, model_dir='notebooks/models')
    model = model.to(device)

    x_train, y_train = _load_dataset(datasets.CIFAR10(root='./data', train=True, download=True, transform=PREPROCESSINGS[None]), n_examples=n_train_examples)
    x_test, y_test = _load_dataset(datasets.CIFAR10(root='./data', train=False, download=True, transform=PREPROCESSINGS[None]), n_examples=n_test_examples)

    ########################
    # Create adversarial examples
    ########################

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.loss = 'ce'
    adversary.apgd.seed = adversary.get_seed()
    adv_curr = adversary.apgd.perturb(x_test, y_test).detach()

    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    ########################
    # Logging
    ########################

    logger.debug(f"Training data: {len(x_train)} samples")
    logger.debug(f"Clean test data: {len(x_test)} samples")
    logger.debug(f"Adversarial test data: {len(adv_curr)} samples")

    return Task.from_separate_data(
        model=model,
        trusted_data=CombinedDataset(x_train, y_train),
        clean_test_data=CombinedDataset(x_test.to(device), y_test),
        anomalous_test_data=CombinedDataset(adv_curr, y_test),
    )