import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from cupbearer import tasks
from cupbearer.analysis.helpers import TaskData
from cupbearer.analysis.variances import plot_variances
from cupbearer.detectors.feature_processing import get_last_token_activation_function_for_task

import csv

def main(dataset, model_name, first_layer, last_layer, base_model='Mistral-7B-v0.1'):
    n_layers = 8
    interval = max(1, (last_layer - first_layer) // n_layers)
    layers = list(range(first_layer, last_layer + 1, interval))

    task = tasks.quirky_lm(
        base_model=base_model,
        include_untrusted=True,
        mixture=True,
        standardize_template=True,
        dataset=dataset,
        random_names=True,
        max_split_size=4000
    )

    activation_processing_function = get_last_token_activation_function_for_task(task)

    layer_list = [f"hf_model.model.layers.{layer}.input_layernorm.input" for layer in layers]

    task_data = TaskData.from_task(
        task,
        activation_names=layer_list,
        activation_processing_func=activation_processing_function,
        n_samples=1000,
        batch_size=4
    )

    fig, average_within_between = plot_variances(task_data, title=f"Variance decomposition for {dataset} task")

    save_dir = Path(f"plots/variances/{base_model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{dataset}_variance_decomposition.png", bbox_inches='tight')
    plt.close(fig)
    
    csv_file = save_dir / "variance_ratios.csv"
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["layer", "variance_ratio", "dataset", "base_model"])
        for layer, ratio in average_within_between.items():
            writer.writerow([layer, ratio, dataset, base_model])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot variance decomposition for tasks")
    parser.add_argument('--model_name', type=str, default='', help='Name of the model to use')
    parser.add_argument('--first_layer', type=int, default=1, help='First layer to use')
    parser.add_argument('--last_layer', type=int, default=31, help='Last layer to use')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset to use')
    parser.add_argument('--base_model', type=str, default='Mistral-7B-v0.1', choices=['Mistral-7B-v0.1', 'Meta-Llama-3.1-8B', 'Meta-Llama-3-8B'], help='Base model to use')

    args = parser.parse_args()

    datasets = [
        "capitals", "hemisphere", "population", "sciq", "sentiment", "nli",
        "authors", "addition", "subtraction", "multiplication",
        "modularaddition", "squaring",
    ]

    if args.dataset == "all":
        for dataset in datasets:
            main(dataset, args.model_name, args.first_layer, args.last_layer, args.base_model)
    else:
        main(args.dataset, args.model_name, args.first_layer, args.last_layer, args.base_model)