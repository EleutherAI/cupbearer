import os
import re
import logging
import numpy as np
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def safe_logprob_to_logit(logprob: float, name: str) -> float:
    """
    Convert a log-probability to a logit value in a safe manner.
    """
    try:
        logit = -logprob - np.log1p(-np.exp(-logprob))
        if np.isinf(logit) or np.isnan(logit):
            logger.warning(f"{name} produced an invalid logit value")
            return float("nan")
        return logit
    except Exception as e:
        logger.warning(f"Error converting {name}: {str(e)}")
        return float("nan")

def get_basename_or_parent(path: str) -> str:
    """
    Return the basename unless it starts with 'all', in which case return the parent folder's basename.
    """
    basename = os.path.basename(path)
    if basename.split("-")[0] != "all":
        return basename
    return os.path.basename(os.path.dirname(path))

def get_layer(layer_tag: str) -> int:
    """
    Parse the layer number from a given tag. Returns:
      -1 for 'all', 32 for 'out', or the first found digit otherwise.
    """
    if layer_tag == "all":
        return -1
    if layer_tag == "out":
        return 32
    numbers = re.findall(r"\d+", layer_tag)
    return int(numbers[0]) if numbers else -1

def get_project_root() -> Path:
    """
    Find the project root directory by looking for specific markers.
    Returns the path to the project root directory.
    """
    current = Path(os.getcwd())
    while current != current.parent:
        if (current / "src" / "cupbearer").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root directory") 