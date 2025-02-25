import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from typing import Tuple
from cupbearer.scripts.data_loader import get_data, filters
from cupbearer.scripts.config import SCORE_ORDER, ONLINE_SCORE_ORDER, OFFLINE_SCORE_ORDER, MART_LOGS_DIR
import logging
from cupbearer.tasks.quirky_lm import quirky_lm

logger = logging.getLogger(__name__)

# Set LaTeX compatible style globally
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 33,
    "axes.titlesize": 33,
    "axes.labelsize": 33,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
})
sns.set_theme(style="whitegrid", font="serif", rc={"text.usetex": True})

metrics_dict = {
    "auc_roc": "AUROC",
    "auc_roc_agree": "AUROC (Agree)",
    "auc_roc_disagree": "AUROC (Disagree)",
    "auc_roc_train_from_test_all": "AUROC (Train vs Test)",
    "auc_roc_train_from_test_agree": "AUROC (Train vs Test) (Agree)",
    "auc_roc_train_from_test_disagree": "AUROC (Train vs Test) (Disagree)"
}

def ensure_plot_dir(plot_type: str) -> str:
    """Create and return path to plot-specific subdirectory in results."""
    plot_dir = os.path.join("results", "plots", plot_type)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def barplot_by_dataset(df: pd.DataFrame, compare: str = "random_names", disagree: bool = False) -> None:
    """
    Plot a bar chart of AUC-ROC by dataset and a specified comparison variable.
    """
    y_col = "auc_roc_disagree" if disagree else "auc_roc"
    if compare == "random_names":
        compare_title = "Labeling Strategy"
    else:
        compare_title = compare
    title = (f"Mean AUC-ROC by Dataset and {compare_title} (only where Alice/Bob disagree)" 
             if disagree else f"Mean AUC-ROC by Dataset and {compare_title}")
    df = df[(df['layer'].isin([-1, 16])) & (df['score'] == 'mistral-activations-mahalanobis')]
    df.dropna(subset=['dataset'], inplace=True)
    
    # Create a mapping for the random_names values
    df = df.copy()  # Create a copy to avoid modifying the original
    if compare == "random_names":
        df[compare] = df[compare].map({
            0: "Single label per class",
            1: "Many labels per class"
        })
    
    grouped_df = df.groupby(["dataset", compare])[y_col].mean().reset_index().sort_values(by="dataset", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="dataset", y=y_col, hue=compare, data=grouped_df)
    plt.title(title, fontsize = 45)
    plt.xlabel("Dataset", fontsize=33)
    plt.ylabel("Alice vs Bob AUC", fontsize=33)
    plt.legend(title=compare_title, fontsize=25, title_fontsize=25, loc='lower right')
    plt.xticks(rotation=45, fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("barplots")
    filename = f"barplot_compare_{compare}_{'disagree' if disagree else 'all'}.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_auc_roc_by_layer_by_score(df: pd.DataFrame, multilayer: bool = True, disagree: bool = False, type: str = "online") -> so.Plot:
    """
    Plot AUC-ROC curves by layer and score.
    """
    df["layer"] = df["layer"].astype(int)
    df = df[df["score"].isin(SCORE_ORDER)]
    y_col = "auc_roc_disagree" if disagree else "auc_roc"
    title_format = "{}"#"{} (disagree only)" if disagree else "{} (all examples)"
    if multilayer:
        df = df[df["layer"] >= 0]
        df["score"] = df["score"].cat.remove_unused_categories()
        df["title"] = df["score"].astype(str) + " " + title_format
        df = df.sort_values(by=["dataset", "layer", "base_model"])
        
        g = (so.Plot(df, x="layer", y=y_col, color="dataset", marker="dataset")
             .facet(col="score", wrap=3)
             .add(so.Line(), so.Agg(), so.Jitter(x=2))
             .label(x="Layer", y=metrics_dict[y_col], title=title_format.format)
             .theme({
                 "figure.figsize": (12, 3 * (len(df["score"].unique()) + 3) // 4),
                 "axes.labelsize": 18,
                 "axes.titlesize": 18,
                 "xtick.labelsize": 15,
                 "ytick.labelsize": 15,
                 "legend.fontsize": 15,
                 "legend.title_fontsize": 15,
             }))
        
        plot_dir = ensure_plot_dir("auc_layer")
        filename = f"auc_layer_{'multilayer' if multilayer else 'single'}_{'disagree' if disagree else 'all'}_{type}.pdf"
        g.save(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
        return g
    else:
        df = df[df["layer"] < 0]
        df["score"] = df["score"].cat.remove_unused_categories()
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        plot_df = df.sort_values(by=["dataset", "layer"])
        g = sns.barplot(data=plot_df, x="score", y="auc_roc", ax=ax, hue="dataset", dodge=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=30)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylim(df["auc_roc"].min() * 0.9, 1.03)
        ax.set_xlabel(None)
        ax.set_title("AUC-ROC by Dataset for Different Scores", fontsize=33)
        ax.legend(fontsize=30)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("auc_layer")
    filename = f"auc_layer_{'multilayer' if multilayer else 'single'}_{'disagree' if disagree else 'all'}_{type}.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    return g

def logits_hist(prompt: str, labels: str) -> None:
    """
    Plot histograms of logits for a given prompt and label combination.
    """
    cols = {
        ("alice", "alice"): "alice_gt_logits_disagree",
        ("alice", "bob"): "alice_wrong_label_logits_disagree",
        ("bob", "alice"): "bob_gt_logits_disagree",
        ("bob", "bob"): "bob_logits_disagree"
    }
    key = (prompt.lower(), labels.lower())
    if key not in cols:
        logger.warning("Invalid prompt/labels combination")
        return
    logits_col = cols[key]
    df, _ = get_data(filters["none"])
    df_filtered = df[df["dataset"].notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(data=df_filtered, x=logits_col, hue="base_model", element="step", stat="density", common_norm=False, ax=ax1, kde=False)
    ax1.set_title(f"Distribution of {prompt}'s log odds on {labels}'s labels")
    ax1.set_xlabel(f"{prompt}'s log odds on {labels}'s labels")
    ax1.set_ylabel("Density")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    sns.histplot(data=df_filtered, x=logits_col, hue="base_model", element="step", stat="density", common_norm=False, ax=ax2, kde=False)
    ax2.set_title(f"Distribution of {prompt}'s log odds on {labels}'s labels")
    ax2.set_xlabel(f"{prompt}'s log odds on {labels}'s labels")
    ax2.set_ylabel("Density")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("logits_hist")
    filename = f"logits_hist_{prompt.lower()}_{labels.lower()}.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_losses(loss_str: str, title_str: str, y_label: str) -> so.Plot:
    """
    Plot loss-related metrics using seaborn.objects.
    """
    _, accuracy_df = get_data(filters["none"])
    accuracy_df = accuracy_df.groupby(["dataset", "base_model"]).agg({
        "bob_logits": "mean",
        "alice_logits": "mean",
        "bob_gt_logits_disagree": "mean",
        "alice_wrong_label_logits_disagree": "mean"
    }).reset_index()
    accdf = accuracy_df.sort_values(by=["dataset", "base_model"])
    color_map = {True: "red", False: "blue"}
    g = (so.Plot(accdf, x="base_model", y=loss_str, color="dataset", marker="dataset")
         .add(so.Line(), so.Agg())
         .scale(color=color_map)
         .label(x="Base Model", y=y_label, title=title_str))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("losses")
    filename = f"losses_{loss_str}.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    return g

def create_tables(type_: str = "online") -> pd.DataFrame:
    """
    Create and save markdown tables summarizing evaluation metrics.
    
    Returns:
        The combined DataFrame.
    """
    score_order_local = OFFLINE_SCORE_ORDER if type_ == "offline" else ONLINE_SCORE_ORDER
    df, _ = get_data(filters["layerwise_agnostic"], score_order=score_order_local)
    df.loc[df["score"].isin(["rephrase"]), "layer"] = -1
    if type_ == "online":
        mart_df, _ = get_data(filters["none"], log_dir=MART_LOGS_DIR)
        mart_df["score"] = "activations-pca-mahalanobis"
        df_all = pd.concat([df, mart_df])
    else:
        df_all = df
    df_all = df_all.dropna(subset=["score"])
    df_all["score"] = df_all["score"].astype(str).str.split("-").str.join(" ").str.replace("\n", " ").str.replace("_", " ")
    df_all["features"] = df_all["features"].astype(str).str.replace("_", " ")
    best_layers = df_all.groupby(["score", "features", "layer"])["auc_roc"].mean().groupby(level=[0, 1]).idxmax().dropna().apply(lambda x: x[2])
    
    def get_best_layer_scores(group: pd.DataFrame) -> pd.Series:
        try:
            best_layer = best_layers.loc[group.name[1], group.name[2]]
            best_scores = group[group["layer"] == best_layer]
            if best_scores.empty:
                return pd.Series([np.nan, np.nan, np.nan], index=["auc_roc", "auc_roc_agree", "auc_roc_disagree"])
            return best_scores.iloc[0][["auc_roc", "auc_roc_agree", "auc_roc_disagree"]]
        except KeyError:
            return pd.Series([np.nan, np.nan, np.nan], index=["auc_roc", "auc_roc_agree", "auc_roc_disagree"])
    
    best_layer_scores = df_all.groupby(["dataset", "score", "features"]).apply(get_best_layer_scores).reset_index(level=0)
    best_layer_scores.columns = ["dataset", "best_auc_roc", "best_auc_roc_agree", "best_auc_roc_disagree"]
    os.makedirs("results", exist_ok=True)
    for dataset in df["dataset"].unique():
        dataset_df = df_all[df_all["dataset"] == dataset].drop(columns="dataset")
        if len(dataset_df) == 0:
            continue
        mean_scores = dataset_df.groupby(["score", "features"])[["auc_roc", "auc_roc_agree", "auc_roc_disagree"]].mean()
        aggregated_scores = dataset_df.loc[dataset_df["layer"] == -1, ["score", "features", "auc_roc", "auc_roc_agree", "auc_roc_disagree"]].dropna()
        aggregated_scores.columns = ["score", "features", "aggregated_auc_roc", "aggregated_auc_roc_agree", "aggregated_auc_roc_disagree"]
        mean_scores.columns = ["mean_auc_roc", "mean_auc_roc_agree", "mean_auc_roc_disagree"]
        combined_scores = mean_scores.join(best_layer_scores.loc[best_layer_scores["dataset"] == dataset, ["best_auc_roc", "best_auc_roc_agree", "best_auc_roc_disagree"]])
        combined_scores = combined_scores.join(aggregated_scores.groupby(["score", "features"])[["aggregated_auc_roc", "aggregated_auc_roc_agree", "aggregated_auc_roc_disagree"]].mean())
        table = combined_scores.reset_index()
        table["best_layer"] = table.apply(lambda row: best_layers.loc[row["score"], row["features"]], axis=1)
        table["best_layer"] = table["best_layer"].astype(str).replace("-1", "aggregate")
        columns = ["score", "features", "mean_auc_roc", "aggregated_auc_roc", "best_auc_roc", "mean_auc_roc_agree", "aggregated_auc_roc_agree", "best_auc_roc_agree", 
                   "mean_auc_roc_disagree", "aggregated_auc_roc_disagree", "best_auc_roc_disagree", "best_layer"]
        def bold_max(s: pd.Series) -> list:
            is_max = s == s.max()
            return ["**" + f"{v:.3f}" + "**" if is_max.iloc[i] else f"{v:.3f}" for i, v in enumerate(s)]
        for col in columns[2:-1]:
            table[col] = bold_max(table[col])
        markdown_table = table[columns].to_markdown(tablefmt="github", index=False)
        tables_dir = ensure_plot_dir("tables")
        with open(os.path.join(tables_dir, f"{dataset}_{type_}_results.md"), "w") as f:
            f.write(markdown_table)
    overall_mean = df_all[
        ((df_all["layer"] == -1) | 
        ((df_all["layer"] == 16) & ~df_all.groupby(["features", "score"])["layer"].transform(lambda x: -1 in x.values))) &
        ~pd.isna(df_all['dataset'])
    ].groupby(["features", "score"]).agg({
        "auc_roc": "mean",
        "auc_roc_agree": "mean",
        "auc_roc_disagree": "mean",
        "dataset": "nunique"
    }).reset_index()
    overall_mean.rename(columns={
        "dataset": "num_datasets",
        "auc_roc": "mean_auc_roc",
        "auc_roc_agree": "mean_auc_roc_agree",
        "auc_roc_disagree": "mean_auc_roc_disagree"
    }, inplace=True)
    for col in ["mean_auc_roc", "mean_auc_roc_agree", "mean_auc_roc_disagree"]:
        overall_mean[col] = bold_max(overall_mean[col])
    overall_mean_table = overall_mean.to_markdown(tablefmt="github", index=False)
    tables_dir = ensure_plot_dir("tables")
    with open(os.path.join(tables_dir, f"overall_{type_}_results.md"), "w") as f:
        f.write(f"\n\n## Overall Aggregated AUROC by Score and Feature: {type_}\n\n")
        f.write(overall_mean_table)
    
    dataset_scores = df_all[
        (df_all["layer"] == -1) | 
        ((df_all["layer"] == 16) & ~df_all.groupby(["features", "score"])["layer"].transform(lambda x: -1 in x.values) &
         ~df_all.groupby(["features", "score"])["layer"].transform(lambda x: 16 in x.values))
    ]
    
    # Create separate aggregations for each base model
    def model_agg(df, model):
        res = df[df["base_model"] == model].groupby(["dataset"]).agg({
            "auc_roc": ["mean", "max"],
            "score": "nunique"
        }).reset_index()
        return res
    
    mistral_scores = model_agg(dataset_scores, "mistral")
    meta_scores = model_agg(dataset_scores, "meta")
    
    # Rename columns for clarity
    mistral_scores.columns = ["dataset", "mistral_mean_auc", "mistral_best_auc", "mistral_num_scores"]
    meta_scores.columns = ["dataset", "meta_mean_auc", "meta_best_auc", "meta_num_scores"]
    
    # Merge the results
    dataset_mean = pd.merge(mistral_scores, meta_scores[["dataset", "meta_mean_auc", "meta_best_auc"]], 
                          on="dataset", how="outer")
    
    # Format all numeric columns
    for col in ["mistral_mean_auc", "mistral_best_auc", "meta_mean_auc", "meta_best_auc"]:
        dataset_mean[col] = bold_max(dataset_mean[col])
    
    dataset_mean_table = dataset_mean.to_markdown(tablefmt="github", index=False)
    with open(os.path.join(tables_dir, "overall_dataset_results.md"), "w") as f:
        f.write("\n\n## Overall Aggregated AUROC by Dataset\n\n")
        f.write(dataset_mean_table)
    
    return df_all 

def plot_scatter_variance(df: pd.DataFrame) -> None:
    """
    Plot scatter plot of AUC-ROC vs variance ratio.
    """
    filtered_df = df[df["score"].isin(["mistral-activations-mahalanobis", "meta-activations-mahalanobis"])]
    filtered_df.rename(columns={
        "variance_ratio": "between-class variance/total variance",
        'base_model': 'Model',
        'dataset': 'Dataset'
    }, inplace=True)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=filtered_df, x="between-class variance/total variance", y="auc_roc",
                    hue="Model", style="Dataset", s=100, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Between-class variance/total variance (log scale)", fontsize=33)
    plt.ylabel("AUC-ROC", fontsize=33)
    plt.title("AUC-ROC (activations-mahalanobis) vs Normalized Class Separation", fontsize=33)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=15, title_fontsize=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("scatter")
    filename = "scatter_variance_ratio_auc.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_scatter_logit(df: pd.DataFrame) -> None:
    """
    Plot scatter plot of logit difference vs variance ratio.
    """
    filtered_df = df[df["score"].isin(["mistral-activations-mahalanobis", "meta-activations-mahalanobis"])]
    if "logit_difference" not in filtered_df.columns:
        filtered_df["logit_difference"] = filtered_df["bob_logits_disagree"] - filtered_df["alice_wrong_label_logits_disagree"]
    
    def fill_variance_ratio(group):
        if -1 in group["layer"].values:
            geomean_variance = np.exp(np.log(group[group["layer"] != -1]["variance_ratio"]).mean())
            group.loc[group["layer"] == -1, "variance_ratio"] = geomean_variance
        return group
    
    fdf2 = filtered_df.groupby(["base_model", "dataset", "score"]).apply(fill_variance_ratio).reset_index(drop=True)
    fdf2 = fdf2[fdf2["layer"] == -1]
    fdf2["logit_difference"] = fdf2["bob_logits_disagree"] - fdf2["alice_wrong_label_logits_disagree"]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=fdf2, x="variance_ratio", y="logit_difference",
                    hue="base_model", style="dataset", sizes=(20, 200), alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Variance Ratio (log scale)", fontsize=33)
    plt.ylabel("Bob Logits - Alice Wrong Label Logits (Disagree)", fontsize=33)
    plt.title("Logit Difference vs Variance Ratio for Activations-Mahalanobis", fontsize=33)
    plt.legend(title="Base Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=15, title_fontsize=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("scatter")
    filename = "scatter_variance_ratio_logit.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_all_trusted_test_label_balance() -> None:
    """
    Plot a clustered bar chart showing label balance for trusted vs test data across all datasets.
    """
    datasets = ["capitals", "hemisphere", "population", "sciq", "sentiment", 
                "nli", "authors", "addition", "subtraction", "multiplication", "modularaddition", "squaring"]
    alice_trusted_percents = []
    bob_trusted_percents = []
    alice_test_percents = []
    bob_test_percents = []
    dataset_names = []
    
    for ds in datasets:
        try:
            task = quirky_lm(
                dataset=ds,
                random_names=True,
                mixture=True,
                include_untrusted=True,
                fake_model=True,
                standardize_template=True
            )
        except Exception as e:
            print(f"Error loading dataset {ds}: {e}")
            continue
            
        alice_train_df = task.untrusted_train_data.normal_data.hf_dataset.to_pandas()
        bob_train_df = task.untrusted_train_data.anomalous_data.hf_dataset.to_pandas()
        alice_test_df = task.test_data.normal_data.hf_dataset.to_pandas()
        bob_test_df = task.test_data.anomalous_data.hf_dataset.to_pandas()
        
        alice_trusted_true = (alice_train_df["label"] == 1).mean() * 100 if not alice_train_df.empty else 0
        bob_trusted_true = (bob_train_df["label"] == 1).mean() * 100 if not bob_train_df.empty else 0
        alice_test_true = (alice_test_df["label"] == 1).mean() * 100 if not alice_test_df.empty else 0
        bob_test_true = (bob_test_df["label"] == 1).mean() * 100 if not bob_test_df.empty else 0
        
        dataset_names.append(ds)
        alice_trusted_percents.append(alice_trusted_true)
        bob_trusted_percents.append(bob_trusted_true)
        alice_test_percents.append(alice_test_true)
        bob_test_percents.append(bob_test_true)
        
    x = np.arange(len(dataset_names))
    width = 0.2  # narrower bars to fit 4 bars per dataset
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - 1.5*width, alice_trusted_percents, width, label="Alice easy")
    plt.bar(x - 0.5*width, bob_trusted_percents, width, label="Bob easy")
    plt.bar(x + 0.5*width, alice_test_percents, width, label="Alice hard")
    plt.bar(x + 1.5*width, bob_test_percents, width, label="Bob hard")
    
    plt.ylabel("Percentage of ``True'' Labels", fontsize=33)
    plt.title("Label Balance: Trusted vs Test Data", fontsize=33, pad=20)
    plt.xticks(x, dataset_names, rotation=45, ha='right', fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=25, loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_dir = ensure_plot_dir("label_balance")
    filename = "all_label_balance_clustered.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches="tight", dpi=300)
    plt.close()

def format_score_label(score: str, include_model: bool = True) -> str:
    """Format score label for plots."""
    # Extract model and format model string
    model_str = ""
    if "mistral-" in score:
        model_str = "Mistral 7B v0.1"
        score = score.replace("mistral-", "")
    elif "meta-" in score:
        model_str = "Llama 3.1 8B"
        score = score.replace("meta-", "")
    
    # Clean up score string
    score = score.replace("\n", " ")
    score = score.replace("mahalanobis", "Mahalanobis")
    
    # Format final string
    if include_model:
        return f"{score} ({model_str})"
    else:
        return f"{score} AUC"

def plot_score_correlations(df: pd.DataFrame, score1: str, score2: str) -> None:
    """
    Plot correlation between AUC scores for two different scoring methods.
    
    Args:
        df: DataFrame containing the evaluation results
        score1: First score to compare
        score2: Second score to compare
        layer: Layer to compare (default -1 for aggregated)
    """
    score1 = score1.replace("\\n", "\n")
    score2 = score2.replace("\\n", "\n")
    # Filter for specified scores and layer
    plot_df = df[
        (df['score'].isin([score1, score2]))
    ].dropna(subset=['auc_roc'])
    
    # Pivot to get scores side by side
    plot_df = plot_df.pivot(
        index=['dataset', 'base_model', 'layer'],
        columns='score',
        values='auc_roc'
    ).reset_index()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Add reference lines first (so points appear on top)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, zorder=1)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, zorder=1)
    
    # Create scatter plot with alpha transparency and larger points
    sns.scatterplot(
        data=plot_df,
        x=score1,
        y=score2,
        hue='dataset',
        s=150,  # Larger points
        alpha=0.7,  # Some transparency
        zorder=2
    )
    
    # Add diagonal line
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]])
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=1)
    
    # Calculate correlation
    corr = plot_df[score1].corr(plot_df[score2])
    
    # Set formatted labels and title
    plt.xlabel(format_score_label(score1, include_model=False), fontsize=33)
    plt.ylabel(format_score_label(score2, include_model=False), fontsize=33)
    plt.title(f'Correlation between\n{format_score_label(score1)} and\n{format_score_label(score2)}\nr = {corr:.3f}', fontsize=33)
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    
    # Add gridlines for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("correlations")
    filename = f"correlation_{score1.replace('-', '_')}_{score2.replace('-', '_')}.pdf"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_quirky_coefficient(df: pd.DataFrame) -> None:
    """
    Plot quirky coefficient analysis showing how it relates to model behavior.
    """
    # Filter for mahalanobis scores
    df = df[df['score'].astype(str).str.contains("activations-mahalanobis")]
    
    # Only use layer 16 for cases where layer -1 is missing
    df = df[
        ((df['layer'] == -1) | 
        ((df['layer'] == 16) & ~df.groupby(['dataset', 'base_model'])['layer'].transform(lambda x: -1 in x.values)))
    ]
    
    # Prepare the data
    plot_df = df.groupby(['dataset', 'base_model'])['quirky_coefficient'].mean().reset_index()
    plot_df.dropna(inplace=True)
    datasets_with_both = plot_df.groupby('dataset').size()
    datasets_with_both = datasets_with_both[datasets_with_both == 2].index
    plot_df = plot_df[plot_df['dataset'].isin(datasets_with_both)]
    
    # Plot 1: Bar plot
    plt.figure(figsize=(20, 8))
    sns.barplot(
        data=plot_df,
        x='dataset',
        y='quirky_coefficient',
        hue='base_model'
    )
    plt.xticks(rotation=45, ha='right', fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('Dataset', fontsize=33)
    plt.ylabel('Quirky Coefficient', fontsize=33)
    plt.title('Quirky Coefficient by Dataset and Model', fontsize=45)
    plt.legend(fontsize=30, title_fontsize=33)
    plt.tight_layout()

    plot_dir = ensure_plot_dir("quirky_coefficient")
    plt.savefig(os.path.join(plot_dir, "quirky_coefficient_by_dataset.pdf"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot correlation plots separately for each model
    for model, title in [('mistral', 'Mistral'), ('meta', 'Llama')]:
        plt.figure(figsize=(12, 8))
        model_df = df[df['base_model']==model]
        
        sns.scatterplot(
            data=model_df,
            x='quirky_coefficient',
            y='auc_roc',
            style='dataset',
            s=100
        )
        plt.title(f'AUC-ROC vs Quirky Coefficient\n({title})', fontsize=45)
        plt.xlabel('Quirkiness', fontsize=33)
        plt.ylabel('AUC-ROC', fontsize=33)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        
        # Add correlation line
        corr_df = model_df[~pd.isna(model_df['quirky_coefficient'])]
        z = np.polyfit(corr_df['quirky_coefficient'], corr_df['auc_roc'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(corr_df['quirky_coefficient'].min(), corr_df['quirky_coefficient'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
        
        # Add correlation text
        corr = corr_df['quirky_coefficient'].corr(corr_df['auc_roc'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, fontsize=30)
        
        if model == 'meta':
            plt.legend(fontsize=15, title_fontsize=15, loc='lower right')
        plt.tight_layout()
        
        # Save model-specific plot
        plot_dir = ensure_plot_dir("quirky_coefficient")
        plt.savefig(os.path.join(plot_dir, f"quirky_coefficient_correlation_{model}.pdf"), bbox_inches='tight', dpi=300)
        plt.close()