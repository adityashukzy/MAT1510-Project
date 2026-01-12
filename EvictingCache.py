from transformers import DynamicCache
from typing import Any, Optional

class EvictingCache(DynamicCache):
    """
    Extending the dynamic cache but allowing for pop operations. 
    """

    def pop1(self) -> None:
        """Remove the last cached token from every initialized layer (in-place)."""
        for layer in self.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            if layer.keys is None or layer.values is None:
                continue
            if layer.keys.numel() == 0:
                continue

            # Remove last token along seq_len axis (-2)
            # keys/values shape: [batch, heads, seq_len, head_dim]
            layer.keys = layer.keys[..., :-1, :]
            layer.values = layer.values[..., :-1, :]

            # Keep sliding-layer bookkeeping consistent if present
            if hasattr(layer, "cumulative_length"):
                # cumulative_length is logical seq length for sliding layers
                layer.cumulative_length = max(int(layer.cumulative_length) - 1, 0)