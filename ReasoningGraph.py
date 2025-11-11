import torch
import numpy as np
import json
from pathlib import Path
from Token import Token

class ReasoningGraph():
    def __init__(self, tokenizer, metric='entropy', branch_percent=0.2, max_initial_nodes=5, branch_per_level=2):
        self.metric = metric
        self.metric_values = []
        self.probabilities = []
        self.logits = []

        self.tokenizer = tokenizer
        
        self.branch_percent = branch_percent
        self.max_initial_nodes = max_initial_nodes
        self.metric_threshold = float('inf') # So none will be above this
        self.branch_per_level = branch_per_level
        
        self.curr_token = None
        self.first_token = None
        self.forking_depth = 0
        self.max_paths = 2 ** max_initial_nodes

        # So we can create the graph after 
        self.input_ids = None
        self.logits_processor = None 
        self.stopping_criteria = None 
        self.generation_config = None 
        self.max_len = None
        self.model_kwargs = None


    # Clear all of the saved metrics
    def zero_metrics(self):
        """Clear all saved metrics and ensure tensor memory is freed."""
        # Clear lists
        self.metric_values = []
        
        # Clear and free tensor memory
        if hasattr(self, 'probabilities'):
            for p in self.probabilities:
                del p
        if hasattr(self, 'logits'):
            for l in self.logits:
                del l
                
        self.probabilities = []
        self.logits = []

    # Calculate the metric on the token distribution
    def _calculate_metric(self, probs):
        metric_list = ['entropy']
        metric_set = set(metric_list)
        if self.metric not in metric_set:
            raise NotImplementedError('This metric is currently not implemented.')

        # Calculate entropy
        log_probs = torch.log(probs + 1e-12)
        step_entropy = -(probs * log_probs).sum(dim=-1)

        return step_entropy
    
    # Generate a single token
    def _generate_single_token(self,
            model,
            input_ids,
            logits_processor,
            generation_config,
            used_tokens=None,
            **model_kwargs
        ):
        temperature = getattr(generation_config, "temperature", 1.0)
        model_kwargs.setdefault("use_cache", True)

        # Prepare the model inputs
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # Forward pass
        outputs = model(**model_inputs, return_dict=True)

        # Update cache and attention mask automatically
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        # Get logits
        next_token_logits = outputs.logits[:, -1, :]

        # Process logits
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Temperature scaling
        do_sample = bool(getattr(generation_config, "do_sample", False))
        if do_sample and temperature > 0:
            next_token_scores = next_token_scores / temperature

            probs = torch.softmax(next_token_scores, dim=-1)

            # Sample from the distribution
            if used_tokens is not None:
                # Remove used tokens
                probs[:, used_tokens] = 0.0

                # Renormalize probs
                probs = probs / probs.sum()
                
            # Sample from distribution
            next_tokens = torch.multinomial(probs, num_samples=1)

        else:
            # Greedy
            probs = torch.softmax(next_token_scores, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)[:, None]

        # Eval metric
        step_metric = self._calculate_metric(probs)

        return next_tokens, step_metric, probs, next_token_scores, model_kwargs
    
    # Determine max tokens for generation
    def _determine_max_tokens(self, input_ids, generation_config):
        # Set max length
        max_len = getattr(generation_config, "max_length", None)
        if max_len is None:
            max_new = getattr(generation_config, "max_new_tokens", None)
            if max_new is None:
                max_new = 256
            max_len = input_ids.shape[1] + max_new

        return max_len
    
    # Determines the metric threshold to flag token as a 'forking token'
    def _find_metric_threshold(self):
        sorted_metrics = sorted(self.metric_values, reverse=True)
        wanted_idx = int(self.branch_percent * len(sorted_metrics)) - 1
        wanted_idx = max(0, min(wanted_idx, len(sorted_metrics) - 1))

        # Making sure it doesn't create too many nodes
        wanted_idx = min(self.max_initial_nodes-1, wanted_idx)

        # Preventing 0 entropy at cutoff
        while sorted_metrics[wanted_idx] <= 0:
            wanted_idx -= 1

        self.metric_threshold = sorted_metrics[wanted_idx]
        
    # Turns the nodes into the first graph
    def _build_graph_from_generation(self, output_ids):
        for output_id, metric_value in zip(output_ids, self.metric_values):

            is_forking = metric_value >= self.metric_threshold

            # The forking nodes aren't going to hold any information now. Just branching
            if is_forking:
                self.forking_depth += 1

                token = Token(
                    token_id=-1,
                    value='',
                    metric=self.metric,
                    metric_value=metric_value,
                    is_forking_token=is_forking,
                    part_of_response=False
                )

            else:
                token = Token(
                    token_id=output_id.item(),
                    value=self.tokenizer.decode(output_id),
                    metric=self.metric,
                    metric_value=metric_value,
                    is_forking_token=is_forking,
                    part_of_response=True
                )

            # if token is our first token, flag it as such
            if self.curr_token is None:
                self.first_token = token
            else:
                # for `self.curr_token`, `token` is the next token
                self.curr_token.add_next_token(token)
                # for `token`, `self.curr_token` is the previous token
                token.add_prev_token(self.curr_token)

            # then, `token` becomes the `curr_token` for the next token
            self.curr_token = token

            if is_forking:
                token = Token(
                    token_id=output_id.item(),
                    value=self.tokenizer.decode(output_id),
                    metric=self.metric,
                    metric_value=metric_value,
                    is_forking_token=False,
                    part_of_response=True
                )
                self.curr_token.add_next_token(token)
                token.add_prev_token(self.curr_token)
                self.curr_token = token

    # Removes the last token from the KV cache
    def _remove_KV_cache(self, model):
        with torch.inference_mode():
            # The attention mask
            attn = torch.ones_like(self.input_ids, dtype=torch.long)

            # Rebuild model_kwargs
            self.model_kwargs = {"use_cache": True, "attention_mask": attn}
            self.model_kwargs = model._get_initial_cache_position(self.input_ids.shape[1], self.input_ids.device, self.model_kwargs)
            model_inputs = model.prepare_inputs_for_generation(self.input_ids, **self.model_kwargs)

            # Run the model to build the cache back. TODO look into ways we can store then query the old cache. Maybe build our own cache variant
            outputs = model(**model_inputs, return_dict=True)
            self.model_kwargs = model._update_model_kwargs_for_generation(outputs, self.model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)

            # Trying to stop the min length being set
            self.generation_config.min_length = 0
            self.generation_config.min_new_tokens = 0

    # Backtracking to create the full graph structure
    def build_full_graph(self, model):
        if self.curr_token is None:
            print("Please generate tokens before creating the graph")
            return
        
        # To track entropy from forking nodes
        forking_metric = None
        used_tokens = None

        # To prevent the exploration going too deep
        original_metric = self.metric_threshold
        num_paths = 1

        print('')
        print('Starting full graph creation')

        backward = True
        with torch.inference_mode():
            # Keep going until it has backtracked all the way
            while (self.curr_token is not self.first_token) or (self.first_token.is_forking_token and len(self.first_token.next_tokens) < self.branch_per_level) and (num_paths <= self.max_paths):
                # Need to change if we're going forward or backward in the search
                if backward:
                    # Sample and go down this path
                    if self.curr_token.is_forking_token and (len(self.curr_token.next_tokens) < self.branch_per_level) and (num_paths <= self.max_paths):
                        self._remove_KV_cache(model)
                        backward = False
                        forking_metric = self.curr_token.metric_value
                        used_tokens = [token.token_id for token in self.curr_token.next_tokens]
                        num_paths += 1
                        
                        print(f'Exploring at depth: {self.forking_depth}')

                    # Move up the graph
                    else:
                        # Tracking where in the graph
                        if self.curr_token.is_forking_token:
                            self.forking_depth -= 1

                            # Taking the depth reduction off
                            if (self.forking_depth <= self.max_initial_nodes) and (self.metric_threshold != original_metric):
                                self.metric_threshold -= 1

                        else:
                            # Reduce the length of the tokens
                            new_len = self.input_ids.shape[1] - 1
                            self.input_ids = self.input_ids[:, :new_len]

                        self.curr_token = self.curr_token.prev_token
                else:
                    # Generate a token
                    next_tokens, step_metric, probs, next_token_scores, self.model_kwargs = self._generate_single_token(model, 
                                                                                                                        self.input_ids, 
                                                                                                                        self.logits_processor, 
                                                                                                                        self.generation_config,
                                                                                                                        used_tokens=used_tokens,
                                                                                                                        **self.model_kwargs)
                        
                    # The branch after a forking token
                    if forking_metric is not None:
                        token = Token(
                            token_id=next_tokens.item(),
                            value=self.tokenizer.decode(int(next_tokens.item())),
                            metric=self.metric,
                            metric_value=forking_metric,
                            is_forking_token=False,
                            part_of_response=True
                        )

                    else:
                        metric_value = step_metric.detach().to(torch.float32).squeeze().cpu().item()

                        is_forking = metric_value >= self.metric_threshold

                        # Forking tokens get their own node
                        if is_forking:
                            self.forking_depth += 1

                            # Adding the depth reduction
                            if (self.forking_depth > self.max_initial_nodes) and (self.metric_threshold == original_metric):
                                self.metric_threshold += 1

                            token = Token(
                                token_id=-1,
                                value='',
                                metric=self.metric,
                                metric_value=metric_value,
                                is_forking_token=is_forking,
                                part_of_response=False
                            )
                            self.curr_token.add_next_token(token)
                            token.add_prev_token(self.curr_token)
                            self.curr_token = token

                        else:
                            token = Token(
                                token_id=next_tokens.item(),
                                value=self.tokenizer.decode(int(next_tokens.item())),
                                metric=self.metric,
                                metric_value=metric_value,
                                is_forking_token=is_forking,
                                part_of_response=True
                            )


                        # Create the token after
                        if is_forking:
                            token = Token(
                                token_id=next_tokens.item(),
                                value=self.tokenizer.decode(int(next_tokens.item())),
                                metric=self.metric,
                                metric_value=metric_value,
                                is_forking_token=False,
                                part_of_response=True
                            )

                    # Move to the next token
                    self.curr_token.add_next_token(token)
                    token.add_prev_token(self.curr_token)
                    self.curr_token = token

                    # Append token
                    self.input_ids = torch.cat([self.input_ids, next_tokens], dim=-1)

                    # Stop condition
                    if self.stopping_criteria(self.input_ids, None) or self.input_ids.shape[1] >= self.max_len:
                        backward = True

                    forking_metric = None
                    used_tokens = None

        print('Done building the full graph')
    
    # Get sequence of tokens ultimately used for response
    def _get_token_sequence(self):
        """Get the sequence of nodes in generation order."""
        if not self.first_token:
            return []
            
        sequence = []
        current = self.first_token

        # Using dfs
        stack = []
        stack.append(current)
        
        while len(stack) > 0:
            node = stack.pop()
            sequence.append(node)
            
            for next_token in node.next_tokens:
                stack.append(next_token)
        
        return sequence

    # Save ReasoningGraph & other artifacts as files for experiment tracking
    def save(self, save_dir, decoded_sequence=None):
        """
        Save the reasoning graph data to a directory with separate files for:
        - Numerical data (metrics, probability distributions)
        - Graph structure with generation metadata
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save numerical data using numpy
        np.savez_compressed(
            save_dir / 'numerical_data.npz',
            metrics=np.array(self.metric_values),
            probabilities=np.stack([p.numpy() for p in self.probabilities]),
            logits=np.stack([l.numpy() for l in self.logits])
        )

        # 2. Save graph structure and metadata
        def node_to_dict(node, idx, node_to_idx):
            # Represent tokens by their indices in the sequential token path
            if node is None:
                return None
                
            # Safely get previous token index
            prev_token_idx = None
            if node.prev_token is not None and node.prev_token in node_to_idx:
                prev_token_idx = node_to_idx[node.prev_token]
            
            # Safely get next token indices
            next_token_indices = []
            if node.next_tokens:  # This is initialized as [] in Token class
                for next_token in node.next_tokens:
                    if next_token in node_to_idx:
                        next_token_indices.append(node_to_idx[next_token])
            
            return {
                'id': node.id,
                'value': node.value,
                'prev_token': prev_token_idx,
                'next_tokens': next_token_indices,
                'metric_value': node.metric_value,
                'is_forking_token': node.is_forking_token,
                'part_of_response': node.part_of_response
            }

        # Get sequential token generation path
        token_sequence = self._get_token_sequence()
        sequence_data = []

        # Mapping from node -> index
        node_to_idx = {node: i for i, node in enumerate(token_sequence)}

        for i, node in enumerate(token_sequence):
            node_data = {
                'token': decoded_sequence[i] if decoded_sequence is not None and i < len(decoded_sequence) else node.value,
                'metric_value': node.metric_value,
                'is_forking_token': node.is_forking_token,
                'part_of_response': node.part_of_response
            }
            sequence_data.append(node_data)

        # Save full structure keyed by node index (safer than token values which may collide)
        nodes_dict = {str(i): node_to_dict(node, i, node_to_idx) for i, node in enumerate(token_sequence)}

        graph_structure = {
            'metadata': {
                'metric': self.metric,
                'branch_percent': self.branch_percent,
                'metric_threshold': float(self.metric_threshold),
                'response_length': len(token_sequence),
            },
            'final_sequence': sequence_data,
            'graph': {
                'nodes': nodes_dict
            }
        }

        with open(save_dir / 'graph_structure.json', 'w') as f:
            json.dump(graph_structure, f, indent=2)

    # Load ReasoningGraph from saved files
    @classmethod
    def load(cls, save_dir, tokenizer):
        """Load a reasoning graph from saved files."""
        save_dir = Path(save_dir)
        
        # Load numerical data
        numerical_data = np.load(save_dir / 'numerical_data.npz')
        
        # Load graph structure
        with open(save_dir / 'graph_structure.json', 'r') as f:
            graph_data = json.load(f)
            metadata = graph_data['metadata']
            graph_structure = graph_data['graph']

        # Create new instance
        graph = cls(
            tokenizer=tokenizer,
            metric=metadata['metric'],
            branch_percent=metadata['branch_percent'],
        )
        
        # Restore numerical data
        graph.metric_values = numerical_data['metrics'].tolist()
        graph.probabilities = [torch.from_numpy(p) for p in numerical_data['probabilities']]
        graph.logits = [torch.from_numpy(l) for l in numerical_data['logits']]
        graph.metric_threshold = metadata['metric_threshold']

        # Reconstruct graph structure (nodes keyed by index)
        nodes = {}
        # First pass: create nodes
        for idx_str, node_data in graph_structure['nodes'].items():
            token_value = node_data.get('value')
            nodes[idx_str] = Token(
                token_id=node_data.get('id'),
                value=token_value,
                metric=node_data.get('metric', 'entropy'),
                metric_value=node_data.get('metric_value'),
                is_forking_token=node_data.get('is_forking_token', False),
                part_of_response=node_data.get('part_of_response', True)
            )

        # Second pass: connect nodes using prev/next relationships
        for idx_str, node_data in graph_structure['nodes'].items():
            node = nodes[idx_str]
            prev_token_idx = node_data.get('prev_token')
            if prev_token_idx is not None:
                prev_key = str(prev_token_idx)
                if prev_key in nodes:
                    node.add_prev_token(nodes[prev_key])
            
            next_token_indices = node_data.get('next_tokens', [])
            for next_idx in next_token_indices:
                next_key = str(next_idx)
                if next_key in nodes:
                    node.add_next_token(nodes[next_key])

        # Set head node to the first sequence element (index 0) if present
        if graph_structure.get('sequence') and len(graph_structure['sequence']) > 0:
            graph.first_token = nodes.get('0')
        else:
            graph.first_token = None

        return graph
    
    # Making the first pass and collecting the metrics
    def _generate_sequence(self, model, input_ids, logits_processor, stopping_criteria, generation_config, max_len, **model_kwargs):
        # Extracting all of the correct values
        model_kwargs = model._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)

        # First pass
        with torch.inference_mode():
            while input_ids.shape[1] < max_len:
                # Single token gen
                next_tokens, step_metric, probs, logits, model_kwargs = self._generate_single_token(model,
                                                                    input_ids,
                                                                    logits_processor,
                                                                    generation_config,
                                                                    **model_kwargs)

                # Saving as a scalar
                self.metric_values.append(step_metric.detach().to(torch.float32).squeeze().cpu().item())

                # Saving the probability distribution
                self.probabilities.append(probs.detach().to(torch.float32).squeeze().cpu())

                # Saving the logits (pre-softmax)
                self.logits.append(logits.detach().to(torch.float32).squeeze().cpu())

                # Append token
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

                # Stop condition
                if stopping_criteria(input_ids, None):
                    break
        
        return input_ids
    
    # Generate tokens and build the entire reasoning graph (called upstream as custom_generate of model.generate)
    def generate_and_build_graph(
            self,
            model,
            input_ids,
            *,
            logits_processor,
            stopping_criteria,
            generation_config,
            **model_kwargs
        ):
        # zeroing all of the calculated metrics. Restarting the graphs
        self.zero_metrics()
        self.curr_token = None
        self.first_token = None 

        # Find max length and starting input length
        max_len = self._determine_max_tokens(input_ids, generation_config)
        start_len = input_ids.shape[1]

        # Generate sequence as first pass
        output_ids = self._generate_sequence(model, input_ids, logits_processor, stopping_criteria, generation_config, max_len, **model_kwargs)

        # Find threshold for metric on the basis of which to flag tokens as 'forking tokens'
        self._find_metric_threshold()

        # Build graph based on generated sequence
        self._build_graph_from_generation(output_ids[0][start_len:])

        # Saving for the graph creation
        self.input_ids = output_ids
        self.logits_processor = logits_processor 
        self.stopping_criteria = stopping_criteria
        self.generation_config = generation_config
        self.max_len = max_len
        self.model_kwargs = model_kwargs
        
        return output_ids