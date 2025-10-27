import torch
import numpy as np
import json
from pathlib import Path
from GraphNode import GraphNode

class ReasoningGraph():
    def __init__(self, metric_type='entropy', branch_percent=0.2, branch_children=2):
        self.metric_type = metric_type
        self.metrics = []
        self.prob_distributions = []
        self.logits = []
        self.branch_percent = branch_percent
        self.branch_children = branch_children
        self.node_cutoff = float('inf') # So none will be above this
        self.cur_node = None
        self.head_node = None

    # Clear all of the saved metrics
    def zero_metrics(self):
        """Clear all saved metrics and ensure tensor memory is freed."""
        # Clear lists
        self.metrics = []
        
        # Clear and free tensor memory
        if hasattr(self, 'prob_distributions'):
            for p in self.prob_distributions:
                del p
        if hasattr(self, 'logits'):
            for l in self.logits:
                del l
                
        self.prob_distributions = []
        self.logits = []

    # Calculate the metric on the token distribution
    def _calculate_metric(self, probs):
        metric_list = ['entropy']
        metric_set = set(metric_list)
        if self.metric_type not in metric_set:
            raise NotImplementedError('This metric is currently not implemented.')

        # Calculate entropy
        log_probs = torch.log(probs + 1e-12)
        step_entropy = -(probs * log_probs).sum(dim=-1)

        return step_entropy
    
    # Generate a single token
    def _single_token_gen(self,
        model,
        input_ids,
        logits_processor,
        generation_config,
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

            # Sample from the distribution
            probs = torch.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

        else:
            # Greedy
            probs = torch.softmax(next_token_scores, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)[:, None]

        # Eval metric
        step_metric = self._calculate_metric(probs)

        return next_tokens, step_metric, probs, next_token_scores, model_kwargs
    
    # Determine max tokens for generation
    def _max_tokens(self, input_ids, generation_config):
        # Set max length
        max_len = getattr(generation_config, "max_length", None)
        if max_len is None:
            max_new = getattr(generation_config, "max_new_tokens", None)
            if max_new is None:
                max_new = 256
            max_len = input_ids.shape[1] + max_new

        return max_len
    
    # Determines the value to separate reasoning and non reasoning tokens
    def _find_node_cutoff(self):
        sorted_metrics = sorted(self.metrics, reverse=True)
        wanted_idx = int(self.branch_percent * len(sorted_metrics)) - 1
        wanted_idx = max(0, min(wanted_idx, len(sorted_metrics) - 1))

        # Preventing 0 entropy at cutoff
        while sorted_metrics[wanted_idx] <= 0:
            wanted_idx -= 1

        self.node_cutoff = sorted_metrics[wanted_idx]
        
    # Turns the nodes into the first graph
    def _first_pass_into_graph(self, output_ids, start_len):
        token_dist = 0
        gen_ids = output_ids[0][start_len:]

        for i, metric in enumerate(self.metrics):
            # Forking token
            if metric >= self.node_cutoff:
                tmp_node = GraphNode(gen_ids[i].item())

                # Adding the new node
                if self.cur_node is None:
                    self.head_node = tmp_node
                else:
                    self.cur_node.add_child(tmp_node, token_dist)
                    tmp_node.add_parent(self.cur_node, token_dist)
                
                self.cur_node = tmp_node
                token_dist = 0
            else:
                token_dist += 1

    # Print the graph
    def print_reasoning_graph(self):
        pass
    
    def get_token_sequence(self):
        """Get the sequence of nodes in generation order."""
        if not self.head_node:
            return []
            
        sequence = []
        current = self.head_node
        while current:
            sequence.append(current)
            # Follow first child for sequential path
            current = current.children[0] if current.children else None
        return sequence

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
            metrics=np.array(self.metrics),
            prob_distributions=np.stack([p.numpy() for p in self.prob_distributions]),
            logits=np.stack([l.numpy() for l in self.logits])
        )

        # 2. Save graph structure and metadata
        def node_to_dict(node, idx, node_to_idx):
            # Represent parents/children by their indices in the sequential token path
            if node is None:
                return None
            parent_indices = [node_to_idx[p] for p in node.parents] if node.parents else []
            child_indices = [node_to_idx[c] for c in node.children] if node.children else []
            return {
                'token_val': node.token_val,
                'token': decoded_sequence[idx] if decoded_sequence is not None and idx < len(decoded_sequence) else None,
                'parents': parent_indices,
                'children': child_indices,
                'parent_dists': node.parent_dists,
                'children_dists': node.children_dists
            }

        # Get sequential token generation path
        token_sequence = self.get_token_sequence()
        sequence_data = []

        # Mapping from node -> index
        node_to_idx = {node: i for i, node in enumerate(token_sequence)}

        for i, node in enumerate(token_sequence):
            node_data = {
                'token': decoded_sequence[i] if decoded_sequence is not None and i < len(decoded_sequence) else node.token_val,
                'entropy': self.metrics[i] if i < len(self.metrics) else None,
                'is_branch_point': (self.metrics[i] >= self.node_cutoff) if i < len(self.metrics) else False,
                'next_distance': node.children_dists[0] if node.children and len(node.children_dists) > 0 else 0
            }
            sequence_data.append(node_data)

        # Save full structure keyed by node index (safer than token values which may collide)
        nodes_dict = {str(i): node_to_dict(node, i, node_to_idx) for i, node in enumerate(token_sequence)}

        graph_data = {
            'metadata': {
                'metric_type': self.metric_type,
                'branch_percent': self.branch_percent,
                'branch_children': self.branch_children,
                'node_cutoff': float(self.node_cutoff),
            },
            'graph': {
                'nodes': nodes_dict,
                'sequence': sequence_data
            }
        }

        with open(save_dir / 'graph_structure.json', 'w') as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def load(cls, save_dir):
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
            metric_type=metadata['metric_type'],
            branch_percent=metadata['branch_percent'],
            branch_children=metadata['branch_children']
        )
        
        # Restore numerical data
        graph.metrics = numerical_data['metrics'].tolist()
        graph.prob_distributions = [torch.from_numpy(p) for p in numerical_data['prob_distributions']]
        graph.logits = [torch.from_numpy(l) for l in numerical_data['logits']]
        graph.node_cutoff = metadata['node_cutoff']

        # Reconstruct graph structure (nodes keyed by index)
        nodes = {}
        # First pass: create nodes
        for idx_str, node_data in graph_structure['nodes'].items():
            token_repr = node_data.get('token') if node_data.get('token') is not None else node_data.get('token_val')
            nodes[idx_str] = GraphNode(
                token_val=token_repr,
                parents=[],
                children=[],
                parent_dists=node_data.get('parent_dists', []),
                children_dists=node_data.get('children_dists', [])
            )

        # Second pass: connect nodes using parent/child indices
        for idx_str, node_data in graph_structure['nodes'].items():
            node = nodes[idx_str]
            for parent_idx in node_data.get('parents', []):
                parent_key = str(parent_idx)
                if parent_key in nodes:
                    node.parents.append(nodes[parent_key])
            for child_idx in node_data.get('children', []):
                child_key = str(child_idx)
                if child_key in nodes:
                    node.children.append(nodes[child_key])

        # Set head node to the first sequence element (index 0) if present
        if graph_structure.get('sequence') and len(graph_structure['sequence']) > 0:
            graph.head_node = nodes.get('0')
        else:
            graph.head_node = None

        return graph
    
    # Making the first pass and collecting the metrics
    def _first_pass_gen(self, model, input_ids, logits_processor, stopping_criteria, generation_config, max_len, **model_kwargs):
        # Extracting all of the correct values
        model_kwargs = model._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)

        # First pass
        with torch.no_grad():
            while input_ids.shape[1] < max_len:
                # Single token gen
                next_tokens, step_metric, probs, logits, model_kwargs = self._single_token_gen(model,
                                                                    input_ids,
                                                                    logits_processor,
                                                                    generation_config,
                                                                    **model_kwargs)

                # Saving as a scalar
                self.metrics.append(step_metric.detach().to(torch.float32).squeeze().cpu().item())

                # Saving the probability distribution
                self.prob_distributions.append(probs.detach().to(torch.float32).squeeze().cpu())

                # Saving the logits (pre-softmax)
                self.logits.append(logits.detach().to(torch.float32).squeeze().cpu())

                # Append token
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

                # Stop condition
                if stopping_criteria(input_ids, None):
                    break
        
        return input_ids
    
    # Build the entire reasoning graph. Called in generate
    def reasoning_graph_builder(self, model,
                         input_ids,
                         *,
                         logits_processor,
                         stopping_criteria,
                         generation_config,
                         **model_kwargs
    ):
        # zeroing all of the calculated metrics. Restarting the graphs
        self.zero_metrics()
        self.cur_node = None
        self.head_node = None 

        # Find max length and starting input length
        max_len = self._max_tokens(input_ids, generation_config)
        start_len = input_ids.shape[1]

        # First pass storing metrics. Return the first reasoning
        output_ids = self._first_pass_gen(model, input_ids, logits_processor, stopping_criteria, generation_config, max_len, **model_kwargs)

        # Create initial graph
        self._find_node_cutoff()
        self._first_pass_into_graph(output_ids, start_len)
        
        return output_ids