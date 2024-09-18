import argparse
from pathlib import Path
import gc
import os
import torch

from cupbearer import tasks, scripts
from cupbearer.detectors.statistical import MahalanobisDetector, QuantumEntropyDetector, IsoForestDetector, LOFDetector, UMAPMahalanobisDetector, UMAPLOFDetector
from cupbearer.detectors.extractors import AttributionEffectExtractor, ActivationExtractor, ProbeEffectExtractor, MultiExtractor
from cupbearer.detectors.feature_processing import get_last_token_activation_function_for_task, concat_to_single_layer
from cupbearer.detectors.extractors.core import FeatureCache
import gc

datasets = [
    "capitals",
    "hemisphere",
    "population",
    # "sciq",
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
    concat=False,
    mlp_out=False,
    base_model='Mistral-7B-v0.1'
):
    n_layers = 8
    interval = max(1, (last_layer - first_layer) // n_layers)
    layers = list(range(first_layer, last_layer + 1, interval))

    task = tasks.quirky_lm(
        base_model=base_model,
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

    if concat:
        global_processing_function = concat_to_single_layer
    else:
        global_processing_function = None

    def effect_prob_func(out, inputs, name):
        logits = out.logits
        assert logits.ndim == 3
        return activation_processing_function(logits, inputs, name)[:, effect_tokens].diff(1)[:,0]

    cache_path = f"cache/{dataset}-{features}-{base_model}-{model_name}-{first_layer}-{last_layer}"
    if features == "attribution":
        cache_path += f"-{ablation}"
    cache = FeatureCache.load(cache_path + ".pt", device=task.model.device) if Path(cache_path + ".pt").exists() else FeatureCache(device=task.model.device)

    extractors = []
    # feature_groups = {f"layer_{layer}": [] for layer in layers}
    if mlp_out:
        layer_list = [f"hf_model.model.layers.{layer}.mlp" for layer in layers]
    else:
        layer_list = [f"hf_model.model.layers.{layer}.self_attn.o_proj" for layer in layers]
    feature_groups = {k: [] for k in layer_list}

    for feature in features:
        if feature == 'attribution':
            layer_dict = {key: (4096,) for key in layer_list}
            for layer, key in zip(layers, layer_dict.keys()):
                # feature_groups[f"layer_{layer}"].append(key)
                feature_groups[key].append(key)

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
                cache_path=f"cache/{dataset}-activations-{base_model}-{model_name}-{first_layer}-{last_layer}.pt",
                cache=cache,
                global_processing_fn=global_processing_function
            ))

            emb = task.model.hf_model.get_input_embeddings()
            emb.requires_grad_(True)

        elif feature == 'activations':
            if len(features) == 1:
                # Use input_layernorm layers if only activations are selected
                if mlp_out:
                    acts_layer_list = [f"hf_model.model.layers.{layer}.mlp.input" for layer in layers]
                else:
                    acts_layer_list = [f"hf_model.model.layers.{layer}.input_layernorm.input" for layer in layers]
            else:
                acts_layer_list = [l + '.output' for l in layer_list]
            
            for layer, key in zip(layers, acts_layer_list):
                # feature_groups[f"layer_{layer}"].append(key)
                if key not in feature_groups:
                    feature_groups[key] = [key]
                else:
                    feature_groups[key].append(key)

            extractors.append(ActivationExtractor(
                names=acts_layer_list,
                individual_processing_fn=activation_processing_function,
                cache=cache,
                global_processing_fn=global_processing_function
            ))
        elif feature == 'probe':
            for layer, key in zip(layers, layer_list):
                # feature_groups[f"layer_{layer}"].append(key)
                if key not in feature_groups:
                    feature_groups[key] = [key]
                else:
                    feature_groups[key].append(key)
            
            effect_capture_args = {'ablation': ablation, 'model_type': 'transformer', 'head_dim': 128}
            if ablation == 'pcs':
                effect_capture_args['n_pcs'] = 10

            extractors.append(ProbeEffectExtractor(
                probe_layers=layer_list,
                intervention_layers=layer_list,
                output_func=effect_prob_func,
                effect_capture_args=effect_capture_args,
                individual_processing_fn=activation_processing_function,
                trusted_data=task.trusted_data,
                model=task.model,
                cache=cache,
                global_processing_fn=global_processing_function
            ))

            emb = task.model.hf_model.get_input_embeddings()
            emb.requires_grad_(True)

    if concat:
        for ex in extractors:
            ex.feature_names = ['all']


    feature_extractor = MultiExtractor(extractors, feature_groups = feature_groups) if len(extractors) > 1 else extractors[0]

    if score == 'mahalanobis':
        if args.umap:
            detector = UMAPMahalanobisDetector(feature_extractor)
        else:
            detector = MahalanobisDetector(feature_extractor)
    elif score == 'que':
        detector = QuantumEntropyDetector(feature_extractor)
    elif score == 'isoforest':
        detector = IsoForestDetector(feature_extractor)
    elif score == 'lof':
        if args.umap:
            detector = UMAPLOFDetector(feature_extractor)
        else:
            detector = LOFDetector(feature_extractor)
    else:
        raise ValueError(f"Unknown score: {score}")
    detector.set_model(task.model)

    batch_size = 7
    eval_batch_size = 7
    if dataset in ['sciq', 'sentiment']:
        batch_size = 2
        eval_batch_size = 2

    save_path = f"logs/quirky/{dataset}-{score}-{'_'.join(features)}-{base_model}-{model_name}-{first_layer}-{last_layer}-{ablation}"

    if concat:
        save_path += "-concat"

    if not layerwise:
        save_path += "/all"

    if Path(save_path).exists():
        if 'detector.pt' in os.listdir(save_path):
            detector.load_weights(Path(save_path) / "detector")
        else:
            detector.load_weights(Path(save_path) /'all'/ "detector.pt")
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
    parser.add_argument('--first_layer', type=int, default=1, help='First layer to use')
    parser.add_argument('--last_layer', type=int, default=31, help='Last layer to use')
    parser.add_argument('--ablation', type=str, default='mean', choices=['mean', 'zero', 'pcs', 'raw', 'grad_norm'], help='Ablation to use')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset to use')
    parser.add_argument('--layerwise', action='store_true', default=False, help='Evaluate layerwise instead of aggregated')
    parser.add_argument('--nonrandom_names', action='store_true', default=False, help='Avoid randomising names')
    parser.add_argument('--features', type=str, nargs='+', default=['activations'], choices=['activations', 'attribution', 'probe'], help='Features to use')
    parser.add_argument('--score', type=str, default='mahalanobis', choices=['mahalanobis', 'que', 'isoforest', 'lof'], help='Score to use')
    parser.add_argument('--concat', action='store_true', default=False, help='Concatenate features across layers')
    parser.add_argument('--umap', action='store_true', default=False, help='Use UMAP instead of Mahalanobis')
    parser.add_argument('--mlp_out', action='store_true', default=False, help='Use MLP output instead of input')
    parser.add_argument('--base_model', type=str, default='Mistral-7B-v0.1', choices=['Mistral-7B-v0.1', 'Meta-Llama-3.1-8B', 'Meta-Llama-3-8B'], help='Base model to use')

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
                concat=args.concat,
                mlp_out=args.mlp_out,
                base_model=args.base_model  # Add this line
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
            concat=args.concat,
            mlp_out=args.mlp_out,
            base_model=args.base_model  # Add this line
        )