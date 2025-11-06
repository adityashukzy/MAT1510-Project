import json
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from ReasoningGraph import ReasoningGraph

class Experiment:
    def __init__(self):
        """Initialize an experiment run."""
        self.timestamp = None
        self.problems = None
        self.config = None
        self.base_dir = None
        self.results = None
        self.full_graph = False
        self.experiment_file = {}
        
    def setup_new(self, model_name, dataset_name, problems, num_rollouts, temperature, full_graph=False):
        """Create directory structure for new experiment."""
        
        # Build new experiment config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.problems = problems
        self.config = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "num_problems": len(self.problems),
            "num_rollouts": num_rollouts,
            "temperature": temperature,
            "full_graph": full_graph,
            "timestamp": self.timestamp
        }
        self.base_dir = Path("experiments") / f"experiment_{self.timestamp}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.full_graph = True

        # Save experiment config
        self.experiment_file["config"] = self.config

        # Add problems too
        self.experiment_file["problems"] = self.problems
        
        with open(self.base_dir / "experiment.json", "w") as f:
            json.dump(self.experiment_file, f, indent=2)
        
        print(f"Setup new experiment at: {self.base_dir} with config:\n{self.config}")
    
    def load_existing(self, experiment_path):
        """Load config from existing experiment."""

        # Convert to Path object and resolve the full path
        experiment_path = Path(experiment_path)

        # If just a directory name is provided, assume it's in the experiments folder
        if not experiment_path.is_absolute() and not experiment_path.exists():
            experiment_path = Path("experiments") / experiment_path

        # Set base directory
        self.base_dir = experiment_path

        # Load existing experiment config
        experiment_file_path = self.base_dir / "experiment.json"
        if not experiment_file_path.exists():
            raise FileNotFoundError(f"No experiment.json found at {experiment_file_path}")

        with open(experiment_file_path, "r") as f:
            self.experiment_file = json.load(f)

        # Initialize instance from loaded config
        self.config = self.experiment_file.get("config", {})
        self.timestamp = self.config.get("timestamp")
        self.results = self.experiment_file.get("results", {})
        self.full_graph = self.experiment_file.get("full_graph", {})
        self.problems = self.experiment_file.get("problems", {})

        print(f"Loaded existing experiment from: {self.base_dir}")
        print(f"Config: {json.dumps(self.config, indent=2)}")
        print(f"Found {len(self.results)} problem(s) with results")

    def _generate_rollouts(
            self,
            model,
            tokenizer,
            problem, 
            problem_idx,
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
        for rollout_idx in tqdm(range(num_rollouts), desc=f"Problem {problem['index']}"):
            # Create brand-new ReasoningGraph for each rollout
            generator = ReasoningGraph(tokenizer=tokenizer)
            
            # Generate one rollout
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=model_inputs["input_ids"],
                    custom_generate=generator.generate_and_build_graph,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    attention_mask=model_inputs["attention_mask"],
                    temperature=temperature
                )
            
            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids)
            
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
            probabilities_list = [p.tolist() for p in generator.probabilities]
            logits_list = [l.tolist() for l in generator.logits]
            
            # Store rollout data with all computed values
            rollout_data = {
                'rollout_idx': rollout_idx,
                'dataset_index': problem['index'],  # Store actual GSM8K dataset index
                'input_text': input_text,
                'output_text': output_text,
                'decoded_tokens': decoded_tokens,
                'predicted_answer': predicted_answer,
                'ground_truth': problem['ground_truth'],
                'is_correct': is_correct,
                'entropy_sequence': generator.metric_values.copy(),  # List of scalars
                'probabilities': probabilities_list,  # List of probability distributions (vocab_size each)
                'logits': logits_list,  # List of logits (vocab_size each)
                'num_tokens': len(generator.metric_values)
            }

            if self.full_graph:
                generator.build_full_graph(model)
            
            # Save rollout in self.results
            self._save_rollout(problem_idx, rollout_idx, generator, rollout_data)
        
        return rollouts

    def _save_rollout(self, problem_idx, rollout_idx, reasoning_graph, rollout_data):
        """Save a single rollout's data."""
        try:
            # Get actual dataset index
            dataset_index = rollout_data.get("dataset_index", problem_idx)
            
            # Create directory for this rollout using dataset index
            rollout_dir = self.base_dir / f"problem_{dataset_index}" / f"rollout_{rollout_idx}"
            # Pass decoded token sequence (if available) to avoid reloading tokenizer later
            reasoning_graph.save(rollout_dir, decoded_sequence=rollout_data.get('decoded_tokens'))
            
            # Find or create problem entry in results
            problem_key = f"problem_{problem_idx}"
            if "results" not in self.experiment_file:
                self.experiment_file["results"] = {}
            
            if problem_key not in self.experiment_file["results"]:
                # Add problem-level metadata only once
                self.experiment_file["results"][problem_key] = {
                    "problem_idx": problem_idx,
                    "dataset_index": dataset_index,
                    "problem_dataset": self.config["dataset_name"],
                    "input_text": rollout_data.get("input_text"),
                    "ground_truth": rollout_data.get("ground_truth"),
                    "rollouts": []
                }
            
            # Add rollout data
            rollout = {
                "rollout_idx": rollout_idx,
                "output_text": rollout_data.get("output_text"),
                "predicted_answer": rollout_data.get("predicted_answer"),
                "is_correct": rollout_data.get("is_correct", False),
                "num_tokens": rollout_data.get("num_tokens", 0)
            }
            
            self.experiment_file["results"][problem_key]["rollouts"].append(rollout)
            
        except Exception as e:
            print(f"Error saving rollout {rollout_idx} for problem {problem_idx}: {str(e)}")
    
    def conduct_experiment(self, model, tokenizer):
        # Initialize results structure if not exists
        if "results" not in self.experiment_file:
            self.experiment_file["results"] = {}
            
        # Run experiments for all problems
        for problem_idx, problem in enumerate(self.problems):
            print(f"\nProblem {problem_idx + 1}/{len(self.problems)}")
            
            # Generate rollouts for this problem (and save result for each)
            self._generate_rollouts(
                model,
                tokenizer,
                problem,
                problem_idx,
                num_rollouts=self.config["num_rollouts"], 
                temperature=self.config["temperature"]
            )

        with open(self.base_dir / "experiment.json", "w") as f:
            json.dump(self.experiment_file, f, indent=2)

        print("\nExperiment completed! Results saved to:", self.base_dir)