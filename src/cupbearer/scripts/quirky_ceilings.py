import json
import plotly.express as px
import numpy as np
import pandas as pd
from cupbearer.utils.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from cupbearer import tasks, analysis
from cupbearer.detectors.activations import get_last_token_activation_function_for_task
from pathlib import Path
import torch
import gc
import argparse
from sklearn.metrics import accuracy_score
import pdb

def main(dataset, attribution, ablation, mlp):
    datasets = [
        "capitals", "hemisphere", "population", "sciq", "sentiment",
        "nli", "authors", "addition", "subtraction", "multiplication",
        "modularaddition", "squaring"
    ]

    if dataset not in datasets:
        raise ValueError(f"Dataset {dataset} is not in the list of available datasets.")

    results = {}

    # Define layers and activation function
    layers = range(0, 32, 4)
    layers = list(layers) + [31]

    # Setup paths
    output_dir = Path(f"logs/quirky/{dataset}-ceil-splitnames")
    if attribution:
        output_dir = Path(f"logs/quirky/{dataset}-attribution-{ablation}-ceil-splitnames")
    if mlp:
        output_dir = Path(f"logs/quirky/{dataset}-mlp-{ablation}-ceil-splitnames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize task
    task = tasks.quirky_lm(include_untrusted=True, mixture=True, standardize_template=True, random_names=True, dataset=dataset)

    activation_func = get_last_token_activation_function_for_task(task)

    # Collect data
    if attribution:
        layer_dict = {f"hf_model.base_model.model.model.layers.{layer}.self_attn": (4096,) for layer in layers}
        if mlp:
            layer_dict = {f"hf_model.base_model.model.model.layers.{layer}.mlp": (4096,) for layer in layers}
        layer_list = list(layer_dict.keys())
        td = analysis.AttributionTaskData.from_task(
            task, 
            layer_dict, 
            n_samples=1024, 
            activation_processing_func=activation_func, 
            batch_size=1,
            ablation=ablation
        )
        td.activations = td.effects
    else:
        layer_list = [f"hf_model.base_model.model.model.layers.{layer}.input_layernorm.input" for layer in layers]
        td = analysis.TaskData.from_task(task, layer_list, n_samples=512, activation_processing_func=activation_func, batch_size=4)

    all_X = []
    all_X_train = []
    
    for i, layer in enumerate(layer_list):
        # Prepare data for logistic regression
        X = td.activations[layer].cpu().numpy()
        y = td.labels.cpu().numpy()
        all_X.append(X)

        X_train = td.activations_train[layer].cpu().numpy()
        y_train = td.labels_train.cpu().numpy()
        all_X_train.append(X_train)

        input_dim = X.shape[1]

        # Train logistic regression
        cl = Classifier(input_dim=input_dim, device=task.model.device)
        cl.fit(torch.from_numpy(X_train).to(task.model.device), torch.from_numpy(y_train).to(task.model.device))

        # Evaluate model
        y_pred_logits = cl.forward(torch.from_numpy(X).to(task.model.device))
        y_pred_proba = torch.sigmoid(y_pred_logits).detach().cpu().numpy()
        roc_auc = roc_auc_score(y, y_pred_proba)
        score = accuracy_score(y, y_pred_proba.round())

        # Store results
        results[layer] = {'accuracy': score, 'roc_auc': roc_auc}

        # Plot results
        df = pd.DataFrame({'True Labels': y, 'Predictions': y_pred_proba})
        fig = px.scatter(df, y='True Labels', x='Predictions', title=f'Logistic Regression Predictions vs True Labels\n for layer {layer}, {dataset}')
        fig.write_image(output_dir / f"{dataset}_{layer}_scatter_plot.png")

    # Concatenate all layers data
    all_X = np.concatenate(all_X, axis=1)
    all_X_train = np.concatenate(all_X_train, axis=1)

    # Train logistic regression on all data
    cl_all = Classifier(input_dim=all_X.shape[1], device=task.model.device)
    cl_all.fit_cv(torch.from_numpy(all_X_train).to(task.model.device), torch.from_numpy(y_train).to(task.model.device), num_penalties=10)

    # Evaluate model on all data
    y_pred_logits_all = cl_all.forward(torch.from_numpy(all_X).to(task.model.device))
    y_pred_proba_all = torch.sigmoid(y_pred_logits_all).detach().cpu().numpy()
    score_all = accuracy_score(y, y_pred_proba_all.round())
    roc_auc_all = roc_auc_score(y, y_pred_proba_all)

    # Store results for all layers concatenated
    results['all_layers'] = {'accuracy': score_all, 'roc_auc': roc_auc_all}

    # Plot results for all layers concatenated
    df_all = pd.DataFrame({'True Labels': y, 'Predictions': y_pred_proba_all})
    fig_all = px.scatter(df_all, y='True Labels', x='Predictions', title=f'Logistic Regression Predictions vs True Labels for all layers concatenated, {dataset}')
    fig_all.write_image(output_dir / f"{dataset}_all_layers_scatter_plot.png")

    # Save results to JSON
    with open(output_dir / f"{dataset}_results.json", 'w') as f:
        print(results)
        json.dump(results, f)

    # Free memory
    del cl, X, y, X_train, y_train, td, task
    gc.collect()
    torch.cuda.empty_cache()

    auc_per_layer = {}
    result_dirs = Path("logs/quirky").glob("*-ceil")
    for dir in result_dirs:
        result_file = dir / f"{dir.name.split('-')[0]}_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                for layer, metrics in data.items():
                    if layer not in auc_per_layer:
                        auc_per_layer[layer] = []
                    auc_per_layer[layer].append(metrics['roc_auc'])

    auc_df = pd.DataFrame.from_dict(auc_per_layer, orient='index').mean(axis=1).reset_index()
    auc_df.columns = ['Layer', 'AUC']
    fig = px.line(auc_df, x='Layer', y='AUC', title='Layer vs AUC for all datasets')
    fig.write_image("logs/quirky/layer_vs_auc.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run quirky ceilings analysis.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to analyze')
    parser.add_argument('--attribution', action='store_true', help='Whether to use attribution')
    parser.add_argument('--ablation', type=str, default='mean', help='Ablation to use (zero, mean, pcs)')
    parser.add_argument('--mlp', action='store_true', help='Whether to use mlp instead of self_attn')
    args = parser.parse_args()
    main(args.dataset, args.attribution, args.ablation, args.mlp)
