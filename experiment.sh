# Clone git repository
git clone https://github.com/adityashukzy/MAT1510-Project.git

# Move into repository
cd MAT1510-Project

# Switch to 'aditya'
git checkout aditya

# Run experiment
python experiment.py \
  --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --dataset "HuggingFaceH4/MATH-500" \
  --num_problems 1 \
  --num_rollouts 1 \
  --temperature 1.0 \
  --zip_experiments