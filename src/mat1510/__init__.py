"""
MAT1510 Reasoning Graph Analysis Package

A framework for analyzing entropy patterns in LLM reasoning through graph-based generation.
"""

from mat1510.core.reasoning_graph import ReasoningGraph
from mat1510.core.reasoning_token import Token
from mat1510.experiment.runner import Experiment
from mat1510.experiment.utils import (
    extract_answer,
    extract_ground_truth_from_solution,
    verify_answer,
    normalize_math,
    load_problems_from_dataset,
    visualize_tokens,
    create_zip_archive,
)
from mat1510.analysis.metrics import (
    compute_per_step_metrics,
    compute_global_metrics,
)

__all__ = [
    # Core
    "ReasoningGraph",
    "Token",
    # Experiment
    "Experiment",
    # Utils
    "extract_answer",
    "extract_ground_truth_from_solution",
    "verify_answer",
    "normalize_math",
    "load_problems_from_dataset",
    "visualize_tokens",
    "create_zip_archive",
    # Analysis
    "compute_per_step_metrics",
    "compute_global_metrics",
]

__version__ = "0.1.0"
