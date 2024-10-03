from cupbearer import utils
from cupbearer.data import MixedData
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import json
import sklearn.metrics
import numpy as np
from tqdm import tqdm
from loguru import logger

from cupbearer import tasks

def individual_log_loss(y_true, y_pred, labels=[0, 1]):
    return -np.sum(np.eye(len(labels))[y_true.astype(int)] * y_pred, axis=1)


def maybe_auc(y_true, y_scores):
    try:
        return sklearn.metrics.roc_auc_score(y_true, y_scores)
    except ValueError:
        return np.nan

def measure_accuracy(task, batch_size=32, pbar=True, save_path=None, histogram_percentile=95):
    def get_scores(batch):
        inputs = utils.inputs_from_batch(batch)
        encoding = task.model.tokenize(inputs, **task.model.tokenize_kwargs)
        mask = encoding['attention_mask']

        no_token = task.model.tokenizer.encode(' No', add_special_tokens=False)[-1]
        yes_token = task.model.tokenizer.encode(' Yes', add_special_tokens=False)[-1]
        effect_tokens = torch.tensor([no_token, yes_token], dtype=torch.long, device=task.model.device)

        logits = task.model(inputs).logits[..., effect_tokens][range(len(inputs)), mask.sum(dim=1) - 1]
        return torch.nn.functional.log_softmax(logits, dim=1)

    dataset = task.test_data
    assert isinstance(dataset, MixedData), type(dataset)

    dataset.return_labels = ['answer', 'anomaly', 'agreement']

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # For some methods, such as adversarial abstractions, it might matter how
        # normal/anomalous data is distributed into batches. In that case, we want
        # to mix them by default.
        shuffle=True,
    )

    metrics = defaultdict(dict)
    if save_path is not None:
        model_name = Path(save_path).parts[-1]
    assert 0 < histogram_percentile <= 100

    if pbar:
        test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

    scores = []
    anomalies = []
    agreement = []
    answers = []

    # It's important we don't use torch.inference_mode() here, since we want
    # to be able to override this in certain detectors using torch.enable_grad().
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, (new_labels, new_anomaly, new_agreements) = batch
            new_scores = get_scores(inputs).cpu().numpy()
            scores.append(new_scores)
            anomalies.append(new_anomaly)
            agreement.append(new_agreements)
            answers.append(new_labels)
    scores = np.concatenate(scores)
    anomalies = np.concatenate(anomalies).astype(bool)
    agreement = np.concatenate(agreement).astype(bool)
    answers = np.concatenate(answers).astype(bool)

    loss = individual_log_loss(answers, scores)
    bob_loss = individual_log_loss(answers[anomalies], scores[anomalies])
    alice_loss = individual_log_loss(answers[~anomalies], scores[~anomalies])
    
    disagreeing = ~agreement
    bob_gt_loss_disagree = individual_log_loss(~answers[disagreeing & anomalies], scores[disagreeing & anomalies])
    bob_loss_disagree = individual_log_loss(answers[disagreeing & anomalies], scores[disagreeing & anomalies])
    alice_gt_loss_disagree = individual_log_loss(answers[disagreeing & ~anomalies], scores[disagreeing & ~anomalies])
    alice_wrong_label_loss_disagree = individual_log_loss(~answers[disagreeing & ~anomalies], scores[disagreeing & ~anomalies])

    logger.info(f"Overall Loss: {loss.mean():.4f}")
    logger.info(f"Bob Loss: {bob_loss.mean():.4f}")
    logger.info(f"Alice Loss: {alice_loss.mean():.4f}")
    logger.info(f"Bob GT Loss (Disagreeing): {bob_gt_loss_disagree.mean():.4f}")
    logger.info(f"Bob Loss (Disagreeing): {bob_loss_disagree.mean():.4f}")
    logger.info(f"Alice GT Loss (Disagreeing): {alice_gt_loss_disagree.mean():.4f}")
    logger.info(f"Alice Wrong-Label Loss (Disagreeing): {alice_wrong_label_loss_disagree.mean():.4f}")
    
    metrics["Overall_Loss"] = loss.tolist()
    metrics["Bob_Loss"] = bob_loss.tolist()
    metrics["Alice_Loss"] = alice_loss.tolist()
    metrics["Bob_GT_Loss_Disagree"] = bob_gt_loss_disagree.tolist()
    metrics["Bob_Loss_Disagree"] = bob_loss_disagree.tolist()
    metrics["Alice_GT_Loss_Disagree"] = alice_gt_loss_disagree.tolist()
    metrics["Alice_Wrong_Label_Loss_Disagree"] = alice_wrong_label_loss_disagree.tolist()

    scores = scores[:, 1]

    auc_roc = maybe_auc(
        answers,
        scores,
    )
    ap = sklearn.metrics.average_precision_score(
        y_true=answers,
        y_score=scores,
    )
    logger.info(f"AUC_ROC: {auc_roc:.4f}")
    logger.info(f"AP: {ap:.4f}")
    metrics["AUC_ROC"] = auc_roc
    metrics["AP"] = ap

    auc_roc_agree_bob = maybe_auc(
        answers[agreement][anomalies[agreement]],  
        scores[agreement][anomalies[agreement]],
    )
    ap_agree_bob = sklearn.metrics.average_precision_score(
        y_true=answers[agreement][anomalies[agreement]],
        y_score=scores[agreement][anomalies[agreement]],
    )
    logger.info(f"AUC_ROC_AGREE_BOB: {auc_roc_agree_bob:.4f}")
    logger.info(f"AP_AGREE_BOB: {ap_agree_bob:.4f}")
    metrics["AUC_ROC_AGREE_BOB"] = auc_roc_agree_bob
    metrics["AP_AGREE_BOB"] = ap_agree_bob

    auc_roc_agree_alice = maybe_auc(
        answers[agreement][~anomalies[agreement]],
        scores[agreement][~anomalies[agreement]],
    )
    ap_agree_alice = sklearn.metrics.average_precision_score(
        y_true=answers[agreement][~anomalies[agreement]],
        y_score=scores[agreement][~anomalies[agreement]],
    )
    logger.info(f"AUC_ROC_AGREE_ALICE: {auc_roc_agree_alice:.4f}")
    logger.info(f"AP_AGREE_ALICE: {ap_agree_alice:.4f}")
    metrics["AUC_ROC_AGREE_ALICE"] = auc_roc_agree_alice
    metrics["AP_AGREE_ALICE"] = ap_agree_alice

    auc_roc_disagree_bob = maybe_auc(
        answers[~agreement][anomalies[~agreement]],
        scores[~agreement][anomalies[~agreement]],
    )
    ap_disagree_bob = sklearn.metrics.average_precision_score(
        y_true=answers[~agreement][anomalies[~agreement]],
        y_score=scores[~agreement][anomalies[~agreement]],
    )
    logger.info(f"AUC_ROC_DISAGREE_BOB: {auc_roc_disagree_bob:.4f}")
    logger.info(f"AP_DISAGREE_BOB: {ap_disagree_bob:.4f}")
    metrics["AUC_ROC_DISAGREE_BOB"] = auc_roc_disagree_bob
    metrics["AP_DISAGREE_BOB"] = ap_disagree_bob

    auc_roc_disagree_alice = maybe_auc(
        answers[~agreement][~anomalies[~agreement]],
        scores[~agreement][~anomalies[~agreement]],
    )
    ap_disagree_alice = sklearn.metrics.average_precision_score(
        y_true=answers[~agreement][~anomalies[~agreement]],
        y_score=scores[~agreement][~anomalies[~agreement]],
    )
    logger.info(f"AUC_ROC_DISAGREE_ALICE: {auc_roc_disagree_alice:.4f}")
    logger.info(f"AP_DISAGREE_ALICE: {ap_disagree_alice:.4f}")
    metrics["AUC_ROC_DISAGREE_ALICE"] = auc_roc_disagree_alice
    metrics["AP_DISAGREE_ALICE"] = ap_disagree_alice

    if not save_path:
        return metrics

    save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)

    # Everything from here is just saving metrics and creating figures
    # (which we skip if they aren't going to be saved anyway).
    with open(save_path / "eval.json", "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure accuracy for a given task")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset to use (or 'all' for all datasets)")
    parser.add_argument("--base_model", type=str, default="Mistral-7B-v0.1", help="Base model to use")
    args = parser.parse_args()

    datasets = [
        "capitals", "hemisphere", "population", "sciq", "sentiment", "nli",
        "authors", "addition", "subtraction", "multiplication",
        "modularaddition", "squaring"
    ]

    if args.dataset == "all":
        for dataset in datasets:
            task = tasks.quirky_lm(
                base_model=args.base_model,
                include_untrusted=True,
                mixture=True,
                standardize_template=True,
                dataset=dataset,
                random_names=True,
                max_split_size=4000
            )
            metrics = measure_accuracy(task, save_path=f"logs/quirky/accuracy/{dataset}_{args.base_model}")
            print(f"Metrics for {dataset}:")
            print(json.dumps(metrics, indent=2))
    else:
        task = tasks.quirky_lm(
            base_model=args.base_model,
            include_untrusted=True,
            mixture=True,
            standardize_template=True,
            dataset=args.dataset,
            random_names=True,
            max_split_size=4000
        )
        metrics = measure_accuracy(task, save_path=f"logs/quirky/accuracy/{args.dataset}_{args.base_model}")
        print(f"Metrics for {args.dataset}:")
        print(json.dumps(metrics, indent=2))