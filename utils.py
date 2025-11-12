import re
import random
import numpy as np
import zipfile
from pathlib import Path
from datetime import datetime
from IPython.display import HTML
from datasets import load_dataset
from sympy import sympify, N

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

def normalize_math(value):
    """Convert mathematical value (number/string/LaTeX) to float.

    Handles various formats:
    - Integers and floats: 0.25, 1, 2.5
    - Fractions: 1/4, "1/4"
    - LaTeX fractions: \\frac{1}{4}
    - LaTeX roots: \\sqrt{2}, 70 \\sqrt{2}

    Args:
        value: Mathematical expression to normalize

    Returns:
        float: Numerical value
    """
    if isinstance(value, (int, float)):
        return float(value)

    # Convert LaTeX to sympify-compatible format
    s = str(value)
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', s)  # \frac{a}{b} -> ((a)/(b))
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)  # \sqrt{x} -> sqrt(x)
    s = re.sub(r'(\d+)\s*sqrt', r'\1*sqrt', s)  # Add * for implicit multiplication
    s = s.replace('\\', '')  # Remove other backslashes

    return float(N(sympify(s)))

def load_problems_from_dataset(dataset_name='openai/gsm8k', problems='10', split='test'):
    """Load a subset of problems from the provided dataset.

    Args:
        dataset_name: Name of HuggingFace dataset from which to sample problems
        problems: Number of problems to sample (e.g., '10') or range (e.g., '0:10', '10-20')
        split: Dataset split ('train' or 'test')
    """

    # Parse problems parameter to detect range or integer
    if ':' in problems or '-' in problems:
        # Range mode: "0:10" or "0-10"
        use_range = True
        separator = ':' if ':' in problems else '-'
        parts = problems.split(separator)
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {problems}. Use 'start:end' or 'start-end'")
        problem_start = int(parts[0])
        problem_end = int(parts[1])
        print(f"\n\nLoading problems [{problem_start}:{problem_end}) from {dataset_name}:{split}")
    else:
        # Integer mode: "10"
        use_range = False
        num_problems = int(problems)
        problem_start = 0
        problem_end = None
        print(f"\n\nLoading {num_problems} problems from {dataset_name}:{split}")

    if dataset_name == 'openai/gsm8k':
        dataset = load_dataset(dataset_name, "main")
    elif dataset_name == 'HuggingFaceH4/MATH-500':
        dataset = load_dataset(dataset_name, "default")
    elif dataset_name == 'math-ai/aime25':
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name)

    dataset = dataset[split if split in dataset else 'train']
    total_problems = len(dataset)

    # Determine indices based on mode
    if use_range:
        # Use range mode
        end = problem_end if problem_end is not None else total_problems
        if end > total_problems:
            print(f"Warning: problem_end ({end}) exceeds dataset size ({total_problems}). Using {total_problems}.")
            end = total_problems
        if problem_start >= end:
            raise ValueError(f"Invalid range: problem_start ({problem_start}) must be less than problem_end ({end})")
        indices = list(range(problem_start, end))
        print(f"Using range [{problem_start}, {end}), total {len(indices)} problems")
    else:
        # Use random sampling mode
        if num_problems > total_problems:
            print(f"Warning: Requested {num_problems} problems but dataset only has {total_problems}.")
            num_problems = total_problems

        indices = random.sample(range(total_problems), num_problems)

    problems = []
    for idx in indices:
        item = dataset[idx]

        # Find the question key for the problem
        if dataset_name in ['openai/gsm8k']:
            question_key = 'question'
        elif dataset_name in ['HuggingFaceH4/MATH-500', 'math-ai/aime25']:
            question_key = 'problem'
        else:
            question_key = 'problem' if 'problem' in item else None
        
        # Find the solution key for the problem
        if dataset_name in ['openai/gsm8k', 'math-ai/aime25']:
            solution_key = 'answer'
        elif dataset_name in ['HuggingFaceH4/MATH-500']:
            solution_key = 'solution'
        else:
            solution_key = 'solution' if 'solution' in item else None
        
        # Find the ground truth for the problem
        if dataset_name in ['openai/gsm8k']:
            gt_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', item['answer'])
            ground_truth = float(gt_match.group(1)) if gt_match else None
        elif dataset_name in ['HuggingFaceH4/MATH-500', 'math-ai/aime25']:
            ground_truth = item['answer']
        else:
            ground_truth = item.get('answer' if 'answer' in item else None)

        problems.append({
            'index': idx,
            'question': item.get(question_key),
            'full_answer': item.get(solution_key),
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

def create_zip_archive(experiment_dir, job_name=None):
    """Create a zip archive of a specific experiment folder.

    Args:
        experiment_dir: Path to the specific experiment directory to archive
        job_name: Optional job name to include in zip filename for uniqueness

    Returns:
        Path to the created zip file, or None if directory doesn't exist
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.exists():
        print(f"Directory '{experiment_dir}' does not exist.")
        return None

    # Create zip filename based on folder name
    # If job_name is already in the folder name (e.g., "aime25_1_20251111_010136"), use it directly
    # Otherwise, if job_name is provided, prepend it to the timestamp
    if job_name and experiment_path.name.startswith(job_name):
        # Folder already has job name (e.g., "aime25_1_20251111_010136")
        zip_filename = f"{experiment_path.name}.zip"
    elif job_name:
        # Folder has default naming, prepend job name
        # Extract timestamp from experiment folder name (e.g., "experiment_20251111_010136" -> "20251111_010136")
        timestamp = experiment_path.name.replace("experiment_", "")
        zip_filename = f"{job_name}_{timestamp}.zip"
    else:
        # Use default naming: experiment_timestamp.zip
        zip_filename = f"{experiment_path.name}.zip"

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in experiment_path.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(experiment_path.parent)
                print(f"Creating zip archive {zip_filename} at {file_path} for experiment path {experiment_dir}")
                zipf.write(file_path, arcname)

    print(f"Zip archive created successfully: {zip_filename}")
    return zip_filename