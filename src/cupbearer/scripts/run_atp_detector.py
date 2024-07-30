import argparse
from pathlib import Path
import gc

import torch

from cupbearer import tasks, scripts
from cupbearer.detectors.statistical import MahalanobisDetector, QuantumEntropyDetector, IsoForestDetector, LOFDetector
from cupbearer.detectors.extractors import AttributionEffectExtractor, ActivationExtractor, ProbeEffectExtractor, MultiExtractor
from cupbearer.detectors.feature_processing import get_last_token_activation_function_for_task
from cupbearer.detectors.extractors.core import FeatureCache
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
    "squaring",
]


def main(
    dataset,
    first_layer,
    last_layer,
    model_name,
    ablation,
    features,
    score,
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

    activation_processing_function = get_last_token_activation_function_for_task(task)

    def effect_prob_func(out, inputs, name):
        logits = out.logits
        assert logits.ndim == 3
        return activation_processing_function(logits, inputs, name)[:, effect_tokens].diff(1)[:,0]

    cache_path = f"cache/{dataset}-{features}-Mistral-7B-v0.1-{model_name}-{first_layer}-{last_layer}"
    if features == "attribution":
        cache_path += f"-{ablation}"
    cache = FeatureCache.load(cache_path + ".pt", device=task.model.device) if Path(cache_path + ".pt").exists() else FeatureCache(device=task.model.device)

    cache = None

    extractors = []
    feature_groups = {f"layer_{layer}": [] for layer in layers}
    for feature in features:
        if feature == 'attribution':
            layer_dict = {}
            for layer in layers:
                key = f"hf_model.model.layers.{layer}.self_attn"
                layer_dict[key] = (4096,)
                feature_groups[f"layer_{layer}"].append(key)

            effect_capture_args = {'ablation': ablation, 'model_type': 'transformer', 'head_dim': 128}
            if ablation == 'pcs':
                effect_capture_args['n_pcs'] = 10
        
            extractors.append(AttributionEffectExtractor(
                names=list(layer_dict.keys()),
                output_func=effect_prob_func,
                effect_capture_args=effect_capture_args,
                individual_processing_fn=activation_processing_function,
                trusted_data=task.trusted_data,
                model=task.model,
                cache_path=f"cache/{dataset}-activations-Mistral-7B-v0.1-{model_name}-{first_layer}-{last_layer}.pt",
                cache=cache
            ))

            emb = task.model.hf_model.get_input_embeddings()
            emb.requires_grad_(True)

        elif feature == 'activations':
            layer_dict = {}
            for layer in layers:
                key = f"hf_model.model.layers.{layer}.input_layernorm.input"
                layer_dict[key] = (4096,)
                feature_groups[f"layer_{layer}"].append(key)

            extractors.append(ActivationExtractor(
                names=list(layer_dict.keys()),
                individual_processing_fn=activation_processing_function,
                cache=cache
            ))
        elif feature == 'probe':
            layer_dict = {}
            for layer in layers:
                key = f"hf_model.model.layers.{layer}.self_attn"
                layer_dict[key] = (4096,)
                feature_groups[f"layer_{layer}"].append(key)
            
            effect_capture_args = {'ablation': ablation, 'model_type': 'transformer', 'head_dim': 128}
            if ablation == 'pcs':
                effect_capture_args['n_pcs'] = 10

            extractors.append(ProbeEffectExtractor(
                probe_layers=list(layer_dict.keys()),
                intervention_layers=list(layer_dict.keys()),
                output_func=effect_prob_func,
                effect_capture_args=effect_capture_args,
                individual_processing_fn=activation_processing_function,
                trusted_data=task.trusted_data,
                model=task.model,
                cache=cache
            ))

            emb = task.model.hf_model.get_input_embeddings()
            emb.requires_grad_(True)

    feature_extractor = MultiExtractor(extractors, feature_groups = feature_groups) if len(extractors) > 1 else extractors[0]

    if score == 'mahalanobis':
        detector = MahalanobisDetector(feature_extractor)
    elif score == 'que':
        detector = QuantumEntropyDetector(feature_extractor)
    elif score == 'isoforest':
        detector = IsoForestDetector(feature_extractor)
    elif score == 'lof':
        detector = LOFDetector(feature_extractor)
    else:
        raise ValueError(f"Unknown score: {score}")
    detector.set_model(task.model)

    batch_size = 7
    eval_batch_size = 7
    if dataset in ['sciq', 'sentiment']:
        batch_size = 2
        eval_batch_size = 2

    save_path = f"logs/quirky/{dataset}-{score}-{'_'.join(features)}-Mistral_7B_v0.1-{model_name}-{first_layer}-{last_layer}-{ablation}"

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
            layerwise=layerwise,
            shuffle=False
        )
    
    cache.store(cache_path)
    
    del task, detector
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Attribution Mahalanobis Detector")
    parser.add_argument('--model_name', type=str, default='', help='Name of the model to use')
    parser.add_argument('--first_layer', type=int, required=True, help='First layer to use')
    parser.add_argument('--last_layer', type=int, required=True, help='Last layer to use')
    parser.add_argument('--ablation', type=str, default='mean', choices=['mean', 'zero', 'pcs', 'raw', 'grad_norm'], help='Ablation to use')
    parser.add_argument('--dataset', type=str, default='sciq', help='Dataset to use')
    parser.add_argument('--layerwise', action='store_true', default=False, help='Evaluate layerwise instead of aggregated')
    parser.add_argument('--nonrandom_names', action='store_true', default=False, help='Avoid randomising names')
    parser.add_argument('--features', type=str, nargs='+', default=['activations'], choices=['activations', 'attribution', 'probe'], help='Features to use')
    parser.add_argument('--score', type=str, default='mahalanobis', choices=['mahalanobis', 'que', 'isoforest', 'lof'], help='Score to use')

    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in datasets:
            main(
                dataset,
                args.first_layer,
                args.last_layer,
                args.model_name,
                args.ablation,
                args.features,
                args.score,
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
            args.features,
            args.score,
            random_names=not args.nonrandom_names,
            layerwise=args.layerwise,
        )