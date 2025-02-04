import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Callable, List, Dict, Any
from cupbearer.scripts.config import LOGS_DIR, MART_LOGS_DIR, ONLINE_SCORE_ORDER, BASE_MODEL_DICT
from cupbearer.scripts.utils import safe_logprob_to_logit, get_basename_or_parent, get_layer

import logging
logger = logging.getLogger(__name__)

# Filter function type annotation.
FilterFunc = Callable[[str, List[str], List[str]], bool]

filters: Dict[str, FilterFunc] = {
    "rand_vs_nonrand": lambda root, dirs, files: "eval.json" in files and 
         "probe_trajectory" in get_basename_or_parent(root) and 
         ("-rand_retrain" in get_basename_or_parent(root) or "nrand_retrain2" in get_basename_or_parent(root) or "--" in get_basename_or_parent(root)),
    "layerwise_activations": lambda root, dirs, files: "eval.json" in files and len(get_basename_or_parent(root).split("-")) > 4 and "activations" in get_basename_or_parent(root),
    "layerwise_agnostic": lambda root, dirs, files: "eval.json" in files and len(get_basename_or_parent(root).split("-")) > 4 and 
         ("activations" in root or "iterative_rephrase" in root or "attribution" in root or "probe" in root or "misconception" in root or "nflow" in root or "sae" in root),
    "none": lambda root, dirs, files: "eval.json" in files
}

def parse_filepath(root: str) -> Dict[str, Any]:
    """
    Parse the directory name (or its parent if needed) to extract file meta-information.

    Parameters:
        root (str): The directory path containing the file.
    
    Returns:
        A dictionary containing the parsed fields:
          - dataset: str
          - score: str (with modifications based on flags found in the path)
          - features: Optional[str]
          - alpha: Optional[int]
          - random_names: int (0 if non-random, 1 otherwise)
          - base_model: str ("mistral" or "meta")
    """
    basename = get_basename_or_parent(root)
    name_parts = basename.split("-")
    dataset = name_parts[0]
    score = name_parts[1] if len(name_parts) > 1 else ""
    features = name_parts[2] if len(name_parts) > 2 else None
    if features:
        features = features.replace("activations_attribution", "attribution_activations")
    alpha = int(name_parts[-1]) if name_parts[-1].isdigit() else None
    random_names = 0 if "nrand" in basename else 1

    # Adjust score based on features and specific patterns in the path
    if features and features.split("_")[0] == "attribution":
        score = features + "-" + score + "\n" + name_parts[-1]
    if features and features.split("_")[0] == "probe":
        score = features + "-" + score + "\n" + name_parts[-1]
    if features and features.split("_")[0] == "activations":
        score = features + "-" + score
    if "nflow" in root:
        score = "flow-" + score
    if "misconception" in root:
        score = "misconception"
    if "rephrase" in root:
        score = "rephrase"
    if name_parts[-1] == "concat":
        score += f"-{name_parts[-2]}"
    if "ensemble" in root:
        score += "-ensemble"
    if "cifar10" in root and len(name_parts) > 3:
        score += f"\n{name_parts[3]}"
    if "mlp_out" in root:
        score += "-mlp_out"
    if "sae" in root:
        if "scaled_mean_diff" in root:
            score = "sae-diag-mahalanobis"
        if "l0" in root:
            score = "sae-l0"
    base_model = "mistral" if "Meta" not in root else "meta"
    score = f"{base_model}-{score}"
    return {
        "dataset": dataset,
        "score": score,
        "features": features,
        "alpha": alpha,
        "random_names": random_names,
        "base_model": base_model,
    }

def get_data(filter_fn: FilterFunc, train_from_test: bool = False, log_dir: str = LOGS_DIR,
             score_order: List[str] = ONLINE_SCORE_ORDER) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load main evaluation data by walking through directories in log_dir,
    parsing filenames with parse_filepath(), and merging with additional accuracy data.

    Returns:
        A tuple with the main DataFrame and the accuracy DataFrame.
    """
    results = []
    for root, dirs, files in os.walk(log_dir):
        if filter_fn(root, dirs, files):
            main_eval_path = os.path.join(root, "eval.json")
            with open(main_eval_path, "r") as f:
                main_eval = json.load(f)
            # Exclude certain directories based on content in root
            if "probe" in root and ("Mistral" not in root and "Meta" not in root):
                continue
            if "attribution_activations" in root and "Mistral" in root:
                continue

            # Use the new helper function to parse the file path info.
            file_info = parse_filepath(root)
            dataset = file_info["dataset"]
            score = file_info["score"]
            features = file_info["features"]
            alpha = file_info["alpha"]
            rand = file_info["random_names"]
            base_model = file_info["base_model"]

            for key, value in main_eval.items():
                layer = get_layer(key)
                results.append({
                    "alpha": alpha,
                    "dataset": dataset,
                    "score": score,
                    "features": features,
                    "layer": layer,
                    "random_names": rand,
                    "auc_roc": value["AUC_ROC"],
                    "auc_roc_agree": value["AUC_ROC_AGREE"],
                    "auc_roc_disagree": value["AUC_ROC_DISAGREE"],
                    "base_model": base_model
                })

    df = pd.DataFrame(results)
    accuracy_data = []
    accuracy_dir = LOGS_DIR / "accuracy"
    for dataset in df["dataset"].unique():
        for base_model in df["base_model"].unique():
            base_model_path = BASE_MODEL_DICT[base_model]
            accuracy_file = accuracy_dir / f"{dataset}_{base_model_path}" / "eval.json"
            if os.path.exists(accuracy_file):
                with open(accuracy_file, "r") as f:
                    acc_data = json.load(f)
                length = min(len(acc_data["Bob_GT_Loss_Disagree"]), len(acc_data["Alice_Wrong_Label_Loss_Disagree"]))
                for i in range(length):
                    alice_logits_disagree = safe_logprob_to_logit(acc_data["Alice_GT_Loss_Disagree"][i], "alice_logits_disagree")
                    bob_logits_disagree = safe_logprob_to_logit(acc_data["Bob_Loss_Disagree"][i], "bob_logits_disagree")
                    alice_bob_logits_disagree = safe_logprob_to_logit(acc_data["Alice_Wrong_Label_Loss_Disagree"][i], "alice_bob_logits_disagree")
                    bob_alice_logits_disagree = safe_logprob_to_logit(acc_data["Bob_GT_Loss_Disagree"][i], "bob_alice_logits_disagree")
                    if any(np.isnan(x) for x in [alice_logits_disagree, bob_logits_disagree,
                                                 alice_bob_logits_disagree, bob_alice_logits_disagree]):
                        continue
                    quirky_coefficient = np.mean([
                        alice_logits_disagree - bob_alice_logits_disagree,
                        bob_logits_disagree - alice_bob_logits_disagree
                    ])
                    accuracy_data.append({
                        "dataset": dataset,
                        "base_model": base_model,
                        "quirky_coefficient": quirky_coefficient,
                        "bob_gt_logits_disagree": bob_alice_logits_disagree,
                        "alice_wrong_label_logits_disagree": alice_bob_logits_disagree,
                        "alice_gt_logits_disagree": alice_logits_disagree,
                        "bob_logits_disagree": bob_logits_disagree,
                        "alice_loss": acc_data["Alice_Loss"][i],
                        "bob_loss": acc_data["Bob_Loss"][i]
                    })
    accuracy_df = pd.DataFrame(accuracy_data)
    accdf_grouped = accuracy_df.groupby(["dataset", "base_model"]).agg({
        "bob_gt_logits_disagree": lambda x: np.nanmean(x),
        "alice_wrong_label_logits_disagree": lambda x: np.nanmean(x),
        "alice_gt_logits_disagree": lambda x: np.nanmean(x),
        "bob_logits_disagree": lambda x: np.nanmean(x),
        "quirky_coefficient": lambda x: np.nanmean(x),
    }).reset_index()
    df = df.merge(accdf_grouped, on=["dataset", "base_model"], how="left")
    dataset_order = ["population", "nli", "sentiment", "hemisphere", "addition", "subtraction",
                     "multiplication", "modularaddition", "squaring"]
    base_model_order = ["mistral", "meta"]
    df["score"] = pd.Categorical(df["score"], categories=score_order, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df["base_model"] = pd.Categorical(df["base_model"], categories=base_model_order, ordered=True)
    df.drop_duplicates(subset=["dataset", "score", "layer", "base_model"], inplace=True)
    variance_dfs = []
    for base_model, model_name in BASE_MODEL_DICT.items():
        variance_file = os.path.join("..", "plots", "variances", model_name, "variance_ratios.csv")
        if os.path.exists(variance_file):
            var_df = pd.read_csv(variance_file)
            var_df["base_model"] = base_model
            var_df["layer"] = var_df["layer"].apply(lambda x: int(x.split(".")[-3]))
            var_df["variance_ratio"] = var_df["variance_ratio"].apply(lambda x: float(x.split(",")[0].split("(")[-1]))
            var_df["variance_ratio"] = var_df["variance_ratio"].apply(lambda x: 0 if x < 0 else min(x, 10000))
            variance_dfs.append(var_df)
    if variance_dfs:
        variance_df = pd.concat(variance_dfs)
        df = df.merge(variance_df[["base_model", "dataset", "layer", "variance_ratio"]],
                      on=["base_model", "dataset", "layer"],
                      how="left")
        min_var_df = variance_df.groupby(["base_model", "dataset"]).agg({"variance_ratio": "min"}).reset_index()
        accuracy_df = accuracy_df.merge(min_var_df, on=["base_model", "dataset"], how="left")
    df["score"] = pd.Categorical(df["score"], categories=score_order, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df["base_model"] = pd.Categorical(df["base_model"], categories=base_model_order, ordered=True)
    return df, accuracy_df 