from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer

from cupbearer.data import HuggingfaceDataset
from cupbearer.models import HuggingfaceLM
import pdb
from .task import Task


def quirky_dataset(dataset):
    return HuggingfaceDataset(dataset, text_key="statement", label_key="label")


def quirky_lm(
    random_names: bool = False,
    mixture: bool = False,
    device="cuda",
    include_untrusted: bool = False,
    fake_model: bool = False,
    standardize_template: bool = False,
    dataset: str = "sciq",
    easy_quantile: float = 0.25,
    hard_quantile: float = 0.75,
    max_split_size: int = 4000
):
    from elk_generalization.datasets.loader_utils import templatize_quirky_dataset, ALICE_NAMES, BOB_NAMES
    from peft import AutoPeftModelForCausalLM

    ########################
    # Load model and data
    ########################

    mixture_str = "mixture" if mixture else "fixed"
    name_str = "multiname" if random_names else "singlename"
    model_name = f"EleutherAI/Mistral-7B-v0.1-{dataset}-random"

    if standardize_template:
        model_name += "-standardized"
    if random_names:
        model_name += "-many-random-names"

    model = None
    tokenizer = None
    # We might not want to actually load a model if we're getting all activations
    # from a cache anyway.
    if not fake_model:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map=device)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

    dataset_name = dataset

    prefix = "EleutherAI"

    if dataset_name == "sciq":
        raw_dataset = load_dataset(f"{prefix}/quirky_{dataset_name}_extended_raw")
    else:
        raw_dataset = load_dataset(f"{prefix}/quirky_{dataset_name}_raw")

    dataset = templatize_quirky_dataset(
        raw_dataset,
        ds_name=f"quirky_{dataset_name}_raw",
        standardize_templates=standardize_template,
        method="random" if mixture else "first",
        random_names=random_names,
        easy_quantile=easy_quantile,
        hard_quantile=hard_quantile,
        finetune=False
    )

    ########################
    # Create test data
    ########################

    if random_names:
        # True samples with other Alice-like names:
        alice_test = dataset["validation"].filter(
            lambda x: any(name in x["statement"] for name in ALICE_NAMES[4:]) and x["character"] == "Alice" and x["difficulty_quantile"] >= hard_quantile
        )
    else:
        alice_test = dataset["validation"].filter(lambda x: x["character"] == "Alice" and x["difficulty_quantile"] >= hard_quantile)

    if random_names and include_untrusted:
        # If include_untrusted is False, we can just use all Bob samples since training
        # data won't have included any Bob-like names.
        bob_test = dataset["validation"].filter(
            lambda x: all(name not in x["statement"] for name in BOB_NAMES[:4]) and x["character"] == "Bob" and x["difficulty_quantile"] >= hard_quantile
        )
    else:
        bob_test = dataset["validation"].filter(lambda x: x["character"] == "Bob" and x["difficulty_quantile"] >= hard_quantile)

    ########################
    # Create training data
    ########################

    alice = dataset["train"].filter(lambda x: any(name in x["statement"] for name in ALICE_NAMES[:4]) and x["character"] == "Alice" and x["difficulty_quantile"] < easy_quantile)

    # If we're using untrusted data, we need to split off some of the Alice data
    # into untrusted data, and also use Bob training data.
    bob_train = None
    alice_untrusted = None
    if include_untrusted:
        bob_train = dataset["train"].filter(lambda x: any(name in x["statement"] for name in BOB_NAMES[:4]) and x["difficulty_quantile"] < easy_quantile)

        n = len(alice)
        alice_trusted = alice.select(range(n // 2))
        alice_untrusted = alice.select(range(n // 2, n))
    else:
        alice_trusted = alice

    ########################
    # Logging
    ########################

    logger.debug(f"Alice trusted: {len(alice_trusted)} samples")
    logger.debug(f"Alice test: {len(alice_test)} samples")
    logger.debug(f"Bob test: {len(bob_test)} samples")
    if include_untrusted:
        logger.debug(f"Alice untrusted: {len(alice_untrusted)} samples")
        logger.debug(f"Bob untrusted: {len(bob_train)} samples")
    else:
        logger.debug("No untrusted data")

    return Task.from_separate_data(
        model=HuggingfaceLM(model=model, tokenizer=tokenizer, device=device),
        trusted_data=quirky_dataset(alice_trusted.select(range(min(len(alice_trusted), max_split_size)))),
        clean_test_data=quirky_dataset(alice_test.select(range(min(len(alice_test), max_split_size)))),
        anomalous_test_data=quirky_dataset(bob_test.select(range(min(len(bob_test), max_split_size)))),
        clean_untrusted_data=quirky_dataset(alice_untrusted.select(range(min(len(alice_untrusted), max_split_size)))),
        anomalous_untrusted_data=quirky_dataset(bob_train.select(range(min(len(bob_train), max_split_size)))),
        train_test_mix=True
    )
