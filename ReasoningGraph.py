import torch
from GraphNode import GraphNode

class ReasoningGraph():
    def __init__(self, metric_type='entropy', branch_percent=0.2, branch_children=2):
        self.metric_type = metric_type
        self.metrics = []
        self.branch_percent = branch_percent
        self.branch_children = branch_children
        self.node_cutoff = float('inf') # So none will be above this
        self.cur_node = None
        self.head_node = None

    # Clear all of the saved metrics
    def zero_metrics(self):
        self.metrics = []

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

        return next_tokens, step_metric, model_kwargs
    
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
    
    # Making the first pass and collecting the metrics
    def _first_pass_gen(self, model, input_ids, logits_processor, stopping_criteria, generation_config, max_len, **model_kwargs):
        # Extracting all of the correct values
        model_kwargs = model._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)

        # First pass 
        with torch.no_grad():
            while input_ids.shape[1] < max_len:
                # Single token gen
                next_tokens, step_metric, model_kwargs = self._single_token_gen(model, 
                                                                    input_ids, 
                                                                    logits_processor, 
                                                                    generation_config,
                                                                    **model_kwargs)
                
                # Saving as a scalar
                self.metrics.append(step_metric.detach().to(torch.float32).squeeze().cpu().item())

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