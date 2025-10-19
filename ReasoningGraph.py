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
        attention_mask = model_kwargs.get("attention_mask", None)

        # Get the output logits from the model
        logits = model(input_ids, **model_kwargs).logits
        next_token_logits = logits_processor(input_ids, logits[:, -1, :])

        # Temperature scaling
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

        else:
            # Greedy
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)[:, None]

        # Eval metric
        step_metric = self._calculate_metric(probs)

        return next_tokens, step_metric
    
    # Build the entire reasoning graph. Called in generate
    def reasoning_graph_builder(self, model,
                         input_ids,
                         *,
                         logits_processor,
                         stopping_criteria,
                         generation_config,
                         **model_kwargs
    ):
        eos_token_id = generation_config.eos_token_id
        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device) # Unfinished sequences
        attention_mask = model_kwargs.get("attention_mask", None)

        # First pass 
        with torch.no_grad():
            while input_ids.shape[1] < stopping_criteria[0].max_length:
                # Generate the next token
                next_tokens, step_metric = self._single_token_gen(model, input_ids, logits_processor, generation_config, **model_kwargs)
                self.metrics.append(step_metric)

                # Determine what sequences are 
                if eos_token_id is not None:
                    next_tokens = next_tokens * unfinished[:, None] + (
                        (1 - unfinished[:, None].long()) * generation_config.pad_token_id
                    )
                    unfinished = unfinished & (next_tokens.squeeze(-1) != eos_token_id)

                # Producing the output
                input_ids = torch.cat((input_ids, next_tokens), dim=-1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)

                # Stopping critera
                if not unfinished.any():
                    break
        
        return input_ids