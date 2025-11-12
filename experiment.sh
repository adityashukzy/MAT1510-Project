#!/bin/bash
#SBATCH --job-name=aditya
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A4500|RTX_A6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=slurm_logs/experiment_%j.out
#SBATCH --error=slurm_logs/experiment_%j.err

# Exit on error
set -e

# Parse command-line arguments with defaults
MODEL="Qwen/Qwen3-4B-Thinking-2507"
DATASET="yentinglin/aime_2025"
PROBLEMS="10"
NUM_ROLLOUTS=50
TEMPERATURE=1.0
MAX_NEW_TOKENS=2048
STORE_PROBS_LOGITS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --problems)
            PROBLEMS="$2"
            shift 2
            ;;
        --num_rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --store_probs_logits)
            STORE_PROBS_LOGITS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "=========================================="

# Define directories
HOME_RESULTS_DIR="$HOME/mat1510_results"
mkdir -p "$HOME_RESULTS_DIR"
mkdir -p slurm_logs

# Find the latest scratch directory (expires-DATE format)
# This works because we're now running on a compute node where /scratch/ exists
SCRATCH_BASE=$(ls -1d /scratch/scratch-space/expires-* 2>/dev/null | sort | tail -1)
if [ -z "$SCRATCH_BASE" ]; then
    echo "ERROR: No scratch space found at /scratch/scratch-space/"
    echo "Available directories:"
    ls -la /scratch/scratch-space/ 2>&1 || echo "Cannot access /scratch/scratch-space/"
    exit 1
fi

echo "Using scratch base: $SCRATCH_BASE"
SCRATCH_DIR="$SCRATCH_BASE/$USER/mat1510_job_${SLURM_JOB_ID}"

# Create scratch workspace
echo "Creating scratch workspace at: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Configure cache directories to avoid filling home directory
export UV_CACHE_DIR="$SCRATCH_DIR/.uv_cache"
export UV_LINK_MODE="copy"  # Suppress hardlink warnings
export TMPDIR="$SCRATCH_DIR/.tmp"
export HF_HOME="$SCRATCH_DIR/.hf_cache"
mkdir -p "$TMPDIR"
mkdir -p "$HF_HOME"
echo "UV cache set to: $UV_CACHE_DIR"
echo "Temp dir set to: $TMPDIR"
echo "HuggingFace cache set to: $HF_HOME"
# Note: All caches now in scratch (cleaned up after each job)

# Cleanup function to ensure scratch is cleaned up even on failure
cleanup() {
    echo "=========================================="
    echo "Cleaning up scratch space..."
    if [ -d "$SCRATCH_DIR" ]; then
        rm -rf "$SCRATCH_DIR"
        echo "Scratch directory removed: $SCRATCH_DIR"
    fi
    echo "=========================================="
}
trap cleanup EXIT

# Clone repository to scratch
echo "Cloning repository to scratch..."
git clone https://github.com/adityashukzy/MAT1510-Project.git
cd MAT1510-Project
git checkout aditya

# Install UV if not already available
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "UV already installed: $(uv --version)"
fi

# Setup Python virtual environment using UV
echo "Creating virtual environment with UV..."
uv venv

echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from pyproject.toml using UV
# This installs all dependencies listed in [project.dependencies]
echo "Installing dependencies with UV from pyproject.toml..."
uv pip install -e .

# Verify GPU availability
echo "=========================================="
echo "GPU Information:"
python -u -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "=========================================="

# Run experiment
echo "Starting experiment..."
echo "=========================================="

# Execute the Python command
CMD="python -u experiment.py --model \"$MODEL\" --dataset \"$DATASET\" --problems \"$PROBLEMS\" --num_rollouts $NUM_ROLLOUTS --temperature $TEMPERATURE --max_new_tokens $MAX_NEW_TOKENS --job_name \"$SLURM_JOB_NAME\" --zip_experiments"

# Add store_probs_logits flag if enabled
if [ "$STORE_PROBS_LOGITS" = true ]; then
    CMD="$CMD --store_probs_logits"
fi

eval $CMD

# Check if experiment succeeded
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Experiment completed successfully!"

    # Copy zip file to home directory
    echo "Copying results to home directory..."

    # Copy zip file if it exists
    if ls experiment_*.zip 1> /dev/null 2>&1; then
        cp experiment_*.zip "$HOME_RESULTS_DIR/"
        echo "Zip file copied to: $HOME_RESULTS_DIR/"
        ls -lh "$HOME_RESULTS_DIR"/experiment_*.zip
    else
        echo "Warning: No zip file found. Did --zip_experiments flag work?"
    fi

    echo "Results saved to: $HOME_RESULTS_DIR"
    echo "You can access them on apps0 at: $HOME_RESULTS_DIR"
    echo "You can download the zip by running (on your local):"
    echo "  scp adshukla@cs.toronto.edu:mat1510_results/experiment_<timestamp>.zip /Users/adityashukzy/Documents/GitHub/MAT1510-Project/experiments/"

    echo "=========================================="
    echo "Finished at: $(date)"
else
    echo "=========================================="
    echo "Experiment failed!"
    echo "Check the error log for details: slurm_logs/experiment_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi