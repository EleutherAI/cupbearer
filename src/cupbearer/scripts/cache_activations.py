import argparse
import gc
from pathlib import Path
from cupbearer import tasks, detectors
from cupbearer.detectors.activations import get_last_token_activation_function_for_task
import torch

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
    "squaring"
]

def build_cache(dataset, layers, batch_size, first_layer, last_layer, interval):
    print(f"Processing dataset: {dataset}")

    if dataset in ['sciq', 'sentiment']:
        batch_size //= 4

    task = tasks.quirky_lm(include_untrusted=True, mixture=True, standardize_template=True, random_names=True, dataset=dataset, easy_quantile=0.25, hard_quantile=0.75)
    
    activation_processing_function = get_last_token_activation_function_for_task(task)
    
    cache_path = f"cache/{dataset}-{task.model.hf_model.config.name_or_path.split('/')[-1]}-{first_layer}-{last_layer}-{interval}.pt"
    
    # if the cache file exists, skip
    if Path(cache_path).exists():
        del task
        gc.collect()
        torch.cuda.empty_cache()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = detectors.activation_based.CacheBuilder(
        cache_path,
        layers,
        device,
        activation_processing_func=activation_processing_function,
    )
    
    detector.model = task.model
    
    # Assuming the task has train and test data
    detector.train(task.trusted_data, task.untrusted_train_data, save_path=None, batch_size=batch_size)
    detector.eval(task.test_data, batch_size=batch_size)
    print(f"Cache saved to {cache_path}")
    del task, detector
    gc.collect()
    torch.cuda.empty_cache()

def main(first_layer, last_layer, batch_size):
    interval = (last_layer - first_layer)//8
    layers = list(range(first_layer, last_layer + 1, interval))

    layer_list = [f"hf_model.model.layers.{layer}.input_layernorm.input" for layer in layers]
    layer_list += [f"hf_model.model.layers.{layer}.self_attn.output" for layer in layers]

    for dataset in datasets:
        build_cache(dataset, layer_list, batch_size, first_layer, last_layer, interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CacheBuilder for quirky datasets")
    parser.add_argument('--first_layer', type=int, default=0, help='First layer to use')
    parser.add_argument('--last_layer', type=int, default=31, help='Last layer to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')

    args = parser.parse_args()

    main(args.first_layer, args.last_layer, args.batch_size)