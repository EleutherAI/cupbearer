import argparse
from pathlib import Path
import gc

import torch

from cupbearer import tasks, scripts
from cupbearer.detectors.statistical.mahalanobis_detector import MahalanobisDetector
from cupbearer.detectors.extractors.attribution_effect_extractor import AttributionEffectExtractor
from cupbearer.detectors.activations import get_last_token_activation_function_for_task
import gc

datasets = [
    "capitals",
    "hemisphere",
    "population",
    "sciq",
    "sentiment",
    "nli",
    "authors",
    "addition",
    "subtraction",
    "multiplication",
    "modularaddition",
    # "squaring", Not trained yet
]


def main(
    dataset,
    first_layer,
    last_layer,
    model_name,
    ablation,
    random_names=True,
    layerwise=True,
):
    n_layers = 8
    interval = max(1, (last_layer - first_layer) // n_layers)
    layers = list(range(first_layer, last_layer + 1, interval))

    task = tasks.quirky_lm(
        include_untrusted=True,
        mixture=True,
        standardize_template=True,
        dataset=dataset,
        random_names=random_names,
        max_split_size=4000
    )

    no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
    yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
    effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device="cpu")

    def effect_prob_func(logits):
        assert logits.ndim == 3
        probs = logits.softmax(-1)
        return probs[:, -1, effect_tokens].diff(1).sum()

    activation_processing_function = get_last_token_activation_function_for_task(task)

    layer_dict = {f"hf_model.model.layers.{layer}.self_attn": (4096,) for layer in layers}

    effect_capture_args = {'ablation': ablation, 'model_type': 'transformer'}
    if ablation == 'pcs':
        effect_capture_args['n_pcs'] = 10

    feature_extractor = AttributionEffectExtractor(
        activation_names=list(layer_dict.keys()),
        output_func=effect_prob_func,
        effect_capture_args=effect_capture_args,
        individual_processing_fn=activation_processing_function,
        trusted_data=task.trusted_data,
        model=task.model
    )

    detector = MahalanobisDetector(feature_extractor)
    detector.set_model(task.model)

    batch_size = 1
    eval_batch_size = 1

    save_path = f"logs/quirky/{dataset}-mahalanobis-attribution-{model_name}-{first_layer}-{last_layer}-{ablation}"

    if Path(save_path).exists():
        detector.load_weights(Path(save_path) / "detector")
        scripts.eval_detector(
            task, detector, save_path, pbar=True, batch_size=eval_batch_size,
            train_from_test=False, layerwise=layerwise
        )
    else:
        scripts.train_detector(
            task, detector,
            batch_size=batch_size,
            save_path=save_path,
            eval_batch_size=eval_batch_size,
            pbar=True,
            train_from_test=False,
            layerwise=layerwise
        )
    
    del task, detector
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Attribution Mahalanobis Detector")
    parser.add_argument('--model_name', type=str, default='', help='Name of the model to use')
    parser.add_argument('--first_layer', type=int, required=True, help='First layer to use')
    parser.add_argument('--last_layer', type=int, required=True, help='Last layer to use')
    parser.add_argument('--ablation', type=str, default='mean', choices=['mean', 'zero', 'pcs', 'raw'], help='Ablation to use')
    parser.add_argument('--dataset', type=str, default='sciq', help='Dataset to use')
    parser.add_argument('--layerwise', action='store_true', default=False, help='Evaluate layerwise instead of aggregated')
    parser.add_argument('--nonrandom_names', action='store_true', default=False, help='Avoid randomising names')

    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in datasets:
            main(
                dataset,
                args.first_layer,
                args.last_layer,
                args.model_name,
                args.ablation,
                random_names=not args.nonrandom_names,
                layerwise=args.layerwise,
            )
    else:
        main(
            args.dataset,
            args.first_layer,
            args.last_layer,
            args.model_name,
            args.ablation,
            random_names=not args.nonrandom_names,
            layerwise=args.layerwise,
        )