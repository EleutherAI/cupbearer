import argparse
from pathlib import Path
import gc

import torch

from cupbearer import tasks
from cupbearer.detectors.extractors import AttributionEffectExtractor, ActivationExtractor, ProbeEffectExtractor
from cupbearer.detectors.feature_processing import get_last_token_activation_function_for_task
from cupbearer.detectors.extractors.core import FeatureCache
from cupbearer.detectors.visualization import FeatureVisualizer

datasets = [
    "capitals", "hemisphere", "population", 
    "sciq", "sentiment", "nli",
    "authors", "addition", "subtraction", "multiplication",
    "modularaddition", "squaring",
]

def main(
    dataset,
    first_layer,
    last_layer,
    model_name,
    ablation,
    features,
    random_names=True,
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

    if features == 'attribution':
        layer_dict = {f"hf_model.model.layers.{layer}.self_attn": (4096,) for layer in layers}

        effect_capture_args = {'ablation': ablation, 'model_type': 'transformer', 'head_dim': 128}
        if ablation == 'pcs':
            effect_capture_args['n_pcs'] = 10
    
        feature_extractor = AttributionEffectExtractor(
            names=list(layer_dict.keys()),
            output_func=effect_prob_func,
            effect_capture_args=effect_capture_args,
            individual_processing_fn=activation_processing_function,
            trusted_data=task.trusted_data,
            model=task.model,
            cache_path=f"cache/{dataset}-activations-Mistral-7B-v0.1-{model_name}-{first_layer}-{last_layer}.pt",
            cache=cache
        )

        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)
    elif features == 'activations':
        layer_dict = {f"hf_model.model.layers.{layer}.input_layernorm.input": (4096,) for layer in layers}

        feature_extractor = ActivationExtractor(
            names=list(layer_dict.keys()),
            individual_processing_fn=activation_processing_function,
            cache=cache
        )
        feature_extractor.set_model(task.model)
    elif features == 'probe':
        layer_dict = {f"hf_model.model.layers.{layer}.self_attn": (4096,) for layer in layers}
        
        effect_capture_args = {'ablation': ablation, 'model_type': 'transformer', 'head_dim': 128}
        if ablation == 'pcs':
            effect_capture_args['n_pcs'] = 10

        feature_extractor = ProbeEffectExtractor(
            probe_layers=list(layer_dict.keys()),
            intervention_layers=list(layer_dict.keys()),
            output_func=effect_prob_func,
            effect_capture_args=effect_capture_args,
            individual_processing_fn=activation_processing_function,
            trusted_data=task.trusted_data,
            model=task.model,
            cache=cache
        )

        emb = task.model.hf_model.get_input_embeddings()
        emb.requires_grad_(True)

    visualizer = FeatureVisualizer(feature_extractor)
    
    save_path = f"visualizations/quirky/{dataset}-{features}-Mistral_7B_v0.1-{model_name}-{first_layer}-{last_layer}-{ablation}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    batch_size = 2 if dataset in ['sciq', 'sentiment'] else 4

    visualizer.train_and_visualize(task, data_types=['trusted', 'test'], use_densmap=False, save_dir=save_path, batch_size=batch_size)
        
    del task, visualizer
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize ATP Features")
    parser.add_argument('--model_name', type=str, default='', help='Name of the model to use')
    parser.add_argument('--first_layer', type=int, required=True, help='First layer to use')
    parser.add_argument('--last_layer', type=int, required=True, help='Last layer to use')
    parser.add_argument('--ablation', type=str, default='mean', choices=['mean', 'zero', 'pcs', 'raw', 'grad_norm'], help='Ablation to use')
    parser.add_argument('--dataset', type=str, default='sciq', help='Dataset to use')
    parser.add_argument('--nonrandom_names', action='store_true', default=False, help='Avoid randomising names')
    parser.add_argument('--features', type=str, default='activations', help='Features to use')

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
                random_names=not args.nonrandom_names,
            )
    else:
        main(
            args.dataset,
            args.first_layer,
            args.last_layer,
            args.model_name,
            args.ablation,
            args.features,
            random_names=not args.nonrandom_names,
        )