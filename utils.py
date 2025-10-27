import re
import random
import numpy as np
from IPython.display import HTML
from datasets import load_dataset

def extract_answer(text):
    """Extract the final numerical answer from the model output."""
    # Look for patterns like "####" (GSM8K format) or boxed answers

    # Pattern 1: #### X format (GSM8K style)
    boxed_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if boxed_match:
        return float(boxed_match.group(1))

    # Pattern 2: \boxed{X} format
    boxed_match = re.search(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if boxed_match:
        return float(boxed_match.group(1))

    # Pattern 3: "The answer is X" or "Therefore, X"
    answer_patterns = [
        r'[Tt]he answer is\s*\$?(-?\d+(?:\.\d+)?)',
        r'[Tt]herefore,?\s*\$?(-?\d+(?:\.\d+)?)',
        r'[Ss]o the answer is\s*\$?(-?\d+(?:\.\d+)?)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    # Pattern 4: Last number in the text
    numbers = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if numbers:
        return float(numbers[-1])

    return None

def load_problems_from_dataset(dataset='openai/gsm8k', num_problems=10, split='test', seed=None):
    """Load a subset of problems from the provided dataset.
    
    Args:
        dataset: Name of HuggingFace dataset from which to sample problems
        num_problems: Number of problems to sample
        split: Dataset split ('train' or 'test')
        seed: Random seed for reproducibility. If None, sampling will be truly random.
    """
    
    print(f"Loading {num_problems} problems from {dataset}:{split}")
    
    dataset = load_dataset(dataset, "main")
    dataset = dataset[split]
    
    # Only set seed if one is provided
    if seed is not None:
        random.seed(seed)
        print(f"Using fixed seed: {seed}")
    
    # Sample without replacement
    total_problems = len(dataset)
    if num_problems > total_problems:
        print(f"Warning: Requested {num_problems} problems but dataset only has {total_problems}.")
        num_problems = total_problems
    
    indices = random.sample(range(total_problems), num_problems)
    
    # Reset seed if we set it
    if seed is not None:
        random.seed()

    problems = []
    for idx in indices:
        item = dataset[idx]
        # Extract ground truth answer from GSM8K format (after ####)
        gt_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', item['answer'])
        ground_truth = float(gt_match.group(1)) if gt_match else None

        problems.append({
            'index': idx,
            'question': item['question'],
            'full_answer': item['answer'],
            'ground_truth': ground_truth
        })

    return problems

def visualize_tokens(tokens, entropies):
    """Visualize tokens with color intensity based on entropy."""
    entropies = np.array(entropies)
    # Normalize to 0-1 range
    norm_entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
    
    html = '<div style="line-height: 2; font-family: monospace; font-size: 14px;">'
    for token, intensity in zip(tokens, norm_entropies):
        # Escape HTML characters
        token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html += f'<span style="background-color: rgba(100, 150, 255, {intensity:.2f});">{token}</span>'
    html += '</div>'
    
    return HTML(html)