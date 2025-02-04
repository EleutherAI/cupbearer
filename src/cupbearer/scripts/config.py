import os
from typing import Dict, List
from cupbearer.scripts.utils import get_project_root

SCORE_ORDER_BASE: List[str] = [
    "activations-que",
    "activations-mahalanobis",
    "activations-lof",
    "attribution-mahalanobis\nmean",
    "attribution-lof\nmean",
    "probe-mahalanobis\nmean",
    "probe-lof\nmean",
    "flow-mahalanobis",
    "flow-laplace",
    "sae-diag-mahalanobis",
    "sae-l0"
]

SCORE_ORDER: List[str] = ['mistral-' + s for s in SCORE_ORDER_BASE] + ['meta-' + s for s in SCORE_ORDER_BASE]
if "mistral-flow-laplace" in SCORE_ORDER:
    SCORE_ORDER.remove("mistral-flow-laplace")

ONLINE_SCORE_ORDER: List[str] = [s for s in SCORE_ORDER if not any(j in s for j in ["likelihood", "que", "-em"])]
OFFLINE_SCORE_ORDER: List[str] = [s for s in SCORE_ORDER if any(j in s for j in ["likelihood", "que", "-em"])]

LOGS_DIR = get_project_root() / "logs" / "quirky"
MART_LOGS_DIR = get_project_root() / "logs" / "mart_logs" / "quirky" / "pca_results_hard"
IMAGE_DIR = get_project_root() / "logs" / "adv_image"

BASE_MODEL_DICT: Dict[str, str] = {
    "mistral": "Mistral-7B-v0.1",
    "meta": "Meta-Llama-3.1-8B"
} 