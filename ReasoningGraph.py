import torch

class ReasoningGraph():
    def __init__(self, metric_type='entropy'):
        self.metric_type = metric_type
        self.metrics = []

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
    
    # Build the entire reasoning graph. Called in generate
    def reasoning_graph_builder(self, model,
                         input_ids,
                         *,
                         logits_processor,
                         stopping_criteria,
                         generation_config,
                         **model_kwargs
    ):
        # zeroing all of the calculated metrics
        self.zero_metrics()

        # Set max length
        max_len = getattr(generation_config, "max_length", None)
        if max_len is None:
            max_new = getattr(generation_config, "max_new_tokens", None)
            if max_new is None:
                max_new = 256
            max_len = input_ids.shape[1] + max_new

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
                
                self.metrics.append(step_metric)

                # Append token
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

                # Stop condition
                if stopping_criteria(input_ids, None):
                    break
        
        return input_ids