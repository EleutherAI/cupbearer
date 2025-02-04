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

logger = logging.getLogger(__name__)

metrics_dict = {
    "auc_roc": "Alice vs Bob AUC",
    "auc_roc_agree": "Alice vs Bob AUC (Agree)",
    "auc_roc_disagree": "Alice vs Bob AUC (Disagree)",
    "auc_roc_train_from_test_all": "Train names vs Test names AUC",
    "auc_roc_train_from_test_agree": "Train names vs Test names AUC (Agree)",
    "auc_roc_train_from_test_disagree": "Train names vs Test names AUC (Disagree)"
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
    title = f"Mean AUC-ROC by Dataset and {compare} (only where Alice/Bob disagree)" if disagree else f"Mean AUC-ROC by Dataset and {compare}"
    grouped_df = df.groupby(["dataset", compare])[y_col].mean().reset_index()
    grouped_df = grouped_df.sort_values(by="dataset", ascending=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="dataset", y=y_col, hue=compare, data=grouped_df)
    plt.title(title)
    plt.xlabel("Dataset")
    plt.ylabel("Alice vs Bob AUC")
    plt.legend(title=compare)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("barplots")
    filename = f"barplot_compare_{compare}_{'disagree' if disagree else 'all'}.png"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_auc_roc_by_layer_by_score(df: pd.DataFrame, multilayer: bool = True, disagree: bool = False) -> so.Plot:
    """
    Plot AUC-ROC curves by layer and score.
    """
    df["layer"] = df["layer"].astype(int)
    df = df[df["score"].isin(SCORE_ORDER)]
    y_col = "auc_roc_disagree" if disagree else "auc_roc"
    title_format = "{} (disagree only)" if disagree else "{} (all examples)"
    if multilayer:
        df = df[df["layer"] >= 0]
        df["score"] = df["score"].cat.remove_unused_categories()
        df["title"] = df["score"].astype(str) + " " + title_format
        df = df.sort_values(by=["dataset", "layer", "base_model"])
        g = (so.Plot(df, x="layer", y=y_col, color="dataset", marker="dataset")
             .facet(col="score", wrap=3)
             .add(so.Line(), so.Agg(), so.Jitter(x=2))
             .label(x="Layer", y=metrics_dict[y_col], title=title_format.format)
             .theme({"figure.figsize": (12, 4 * (len(df["score"].unique()) + 3) // 3)}))
    else:
        df = df[df["layer"] < 0]
        df["score"] = df["score"].cat.remove_unused_categories()
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        plot_df = df.sort_values(by=["dataset", "layer"])
        g = sns.barplot(data=plot_df, x="score", y="auc_roc", ax=ax, hue="dataset", dodge=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylim(df["auc_roc"].min() * 0.9, 1.03)
        ax.set_xlabel(None)
        ax.set_title("AUC-ROC by Dataset for Different Scores")
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("auc_layer")
    filename = f"auc_layer_{'multilayer' if multilayer else 'single'}_{'disagree' if disagree else 'all'}.png"
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
    filename = f"logits_hist_{prompt.lower()}_{labels.lower()}.png"
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
    filename = f"losses_{loss_str}.png"
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
        mean_scores = dataset_df.groupby(["score", "features"]).agg(np.nanmean)[["auc_roc", "auc_roc_agree", "auc_roc_disagree"]].dropna()
        aggregated_scores = dataset_df.loc[dataset_df["layer"] == -1, ["score", "features", "auc_roc", "auc_roc_agree", "auc_roc_disagree"]].dropna()
        aggregated_scores.columns = ["score", "features", "aggregated_auc_roc", "aggregated_auc_roc_agree", "aggregated_auc_roc_disagree"]
        mean_scores.columns = ["mean_auc_roc", "mean_auc_roc_agree", "mean_auc_roc_disagree"]
        combined_scores = mean_scores.join(best_layer_scores.loc[best_layer_scores["dataset"] == dataset, ["best_auc_roc", "best_auc_roc_agree", "best_auc_roc_disagree"]])
        combined_scores = combined_scores.join(aggregated_scores.groupby(["score", "features"]).agg(np.nanmean)[["aggregated_auc_roc", "aggregated_auc_roc_agree", "aggregated_auc_roc_disagree"]])
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
        with open(os.path.join("results", "tables", f"{dataset}_{type_}_results.md"), "w") as f:
            f.write(markdown_table)
    overall_mean = df_all[df_all["layer"] == -1].groupby(["features", "score"]).agg({
        "auc_roc": "mean",
        "auc_roc_agree": "mean",
        "auc_roc_disagree": "mean"
    }).reset_index()
    overall_mean.columns = ["features", "score", "mean_auc_roc", "mean_auc_roc_agree", "mean_auc_roc_disagree"]
    for col in ["mean_auc_roc", "mean_auc_roc_agree", "mean_auc_roc_disagree"]:
        overall_mean[col] = bold_max(overall_mean[col])
    overall_mean_table = overall_mean.to_markdown(tablefmt="github", index=False)
    with open(os.path.join("results", "tables", f"overall_{type_}_results.md"), "w") as f:
        f.write(f"\n\n## Overall Aggregated AUROC by Score and Feature: {type_}\n\n")
        f.write(overall_mean_table)
    return df_all 

def plot_scatter_variance(df: pd.DataFrame) -> None:
    """
    Plot scatter plot of AUC-ROC vs variance ratio.
    """
    filtered_df = df[df["score"].isin(["mistral-activations-mahalanobis", "meta-activations-mahalanobis"])]
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=filtered_df, x="variance_ratio", y="auc_roc",
                    hue="base_model", style="dataset", sizes=(20, 200), alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Between-class variance/total variance (log scale)")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC (activations-mahalanobis) vs Variance Ratio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("scatter")
    filename = "scatter_variance_ratio_auc.png"
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
    plt.xlabel("Variance Ratio (log scale)")
    plt.ylabel("Bob Logits - Alice Wrong Label Logits (Disagree)")
    plt.title("Logit Difference vs Variance Ratio for Activations-Mahalanobis")
    plt.legend(title="Base Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    # Save plot
    plot_dir = ensure_plot_dir("scatter")
    filename = "scatter_variance_ratio_logit.png"
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
    plt.close() 