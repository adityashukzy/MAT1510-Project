import json
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from ReasoningGraph import ReasoningGraph

class Experiment:
    def __init__(self, model_name, problems, num_rollouts, temperature):
        """Initialize an experiment run."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.problems = problems
        self.config = {
            "model_name": model_name,
            "num_problems": len(self.problems),
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "timestamp": self.timestamp
        }
        self.base_dir = Path("experiments") / f"experiment_{self.timestamp}"
        self.results = []
        self.experiment_file = {}
        
    def setup(self):
        """Create experiment directory structure."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        self.experiment_file["config"] = self.config
        
        with open(self.base_dir / "experiment.json", "w") as f:
            json.dump(self.experiment_file, f, indent=2)
        
        print(f"Created experiment at: {self.base_dir} with config:\n{self.config}")

    def generate_rollouts(
            self,
            model,
            tokenizer,
            reasoning_graph,
            problem, 
            num_rollouts=16,
            temperature=1.0,
            max_new_tokens=2048
        ):
        """Generate multiple rollouts for a single problem."""
        
        # Format the prompt
        prompt = "Answer the following question: {question}".format(question=problem['question'])
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_text = tokenizer.decode(model_inputs['input_ids'][0])
        
        # Generate rollouts
        rollouts = []
        for rollout_idx in tqdm(range(num_rollouts), desc=f"GSM8K Problem {problem['index']}", leave=False):
            # Reset reasoning graph completely for each rollout
            reasoning_graph.zero_metrics()
            reasoning_graph.cur_node = None
            reasoning_graph.head_node = None
            
            # Generate
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=model_inputs["input_ids"],
                    custom_generate=reasoning_graph.reasoning_graph_builder,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    attention_mask=model_inputs["attention_mask"],
                    temperature=temperature
                )
            
            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Decoded tokens (per-token strings)
            decoded_tokens = [tokenizer.decode([tid]) for tid in output_ids]
            
            # Extract predicted answer
            predicted_answer = extract_answer(output_text)
            
            # Check correctness
            is_correct = False
            if predicted_answer is not None and problem['ground_truth'] is not None:
                # Allow small floating point tolerance
                is_correct = abs(predicted_answer - problem['ground_truth']) < 1e-6
            
            # Convert probability distributions and logits to lists for JSON serialization
            # These are CPU tensors already from ReasoningGraph
            prob_distributions_list = [p.tolist() for p in reasoning_graph.prob_distributions]
            logits_list = [l.tolist() for l in reasoning_graph.logits]
            
            # Store rollout data with all computed values
            rollout_data = {
                'rollout_idx': rollout_idx,
                'gsm8k_index': problem['index'],  # Store actual GSM8K dataset index
                'input_text': input_text,
                'output_text': output_text,
                'decoded_tokens': decoded_tokens,
                'predicted_answer': predicted_answer,
                'ground_truth': problem['ground_truth'],
                'is_correct': is_correct,
                'entropy_sequence': reasoning_graph.metrics.copy(),  # List of scalars
                'prob_distributions': prob_distributions_list,  # List of probability distributions (vocab_size each)
                'logits': logits_list,  # List of logits (vocab_size each)
                'num_tokens': len(reasoning_graph.metrics)
            }
            
            rollouts.append(rollout_data)
        
        return rollouts

    def save_rollout(self, problem_idx, rollout_idx, reasoning_graph, rollout_data):
        """Save a single rollout's data."""
        try:
            # Get actual GSM8K dataset index
            gsm8k_idx = rollout_data.get("gsm8k_index", problem_idx)
            
            # Create directory for this rollout using GSM8K index
            rollout_dir = self.base_dir / f"problem_{gsm8k_idx}" / f"rollout_{rollout_idx}"
            # Pass decoded token sequence (if available) to avoid reloading tokenizer later
            reasoning_graph.save(rollout_dir, decoded_sequence=rollout_data.get('decoded_tokens'))
            
            # Calculate statistics safely
            entropy_sequence = rollout_data.get("entropy_sequence", [])
            entropy_mean = (sum(entropy_sequence) / len(entropy_sequence)) if entropy_sequence else 0
            entropy_std = float(np.std(entropy_sequence)) if entropy_sequence else 0
            
            # Add metadata about correctness, answer, etc.
            self.results.append({
                "problem_idx": problem_idx,  # Sequential index in our experiment
                "gsm8k_index": gsm8k_idx,   # Actual index in GSM8K dataset
                "rollout_idx": rollout_idx,
                "input_text": rollout_data.get("input_text"),
                "output_text": rollout_data.get("output_text"),
                "predicted_answer": rollout_data.get("predicted_answer"),
                "ground_truth": rollout_data.get("ground_truth"),
                "is_correct": rollout_data.get("is_correct", False),
                "num_tokens": rollout_data.get("num_tokens", 0),
                "entropy_mean": entropy_mean,
                "entropy_std": entropy_std,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            })
        except Exception as e:
            print(f"Error saving rollout {rollout_idx} for problem {problem_idx}: {str(e)}")
    
    def conduct_experiment(self, model, tokenizer):
        # Run experiments for all problems
        for prob_idx, problem in enumerate(self.problems):
            print(f"\nProblem {prob_idx + 1}/{len(self.problems)}")
            
            # Generate rollouts for this problem
            reasoning_graph = ReasoningGraph()
            rollouts = self.generate_rollouts(
                model,
                tokenizer,
                reasoning_graph, 
                problem, 
                num_rollouts=self.config["num_rollouts"], 
                temperature=self.config["temperature"]
            )
            
            # Save each rollout
            for rollout_idx, rollout_data in enumerate(rollouts):
                self.save_rollout(prob_idx, rollout_idx, reasoning_graph, rollout_data)

        # Save experiment results
        self.experiment_file["results"] = self.results

        with open(self.base_dir / "experiment.json", "w") as f:
            json.dump(self.experiment_file, f, indent=2)

        print("\nExperiment completed! Results saved to:", self.base_dir)