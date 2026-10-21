"""Instance-local SDPA adapter for the pinned Transformers 4.35 Llama.

The projection, rotary embedding and legacy KV tuple semantics are retained.
No BLIP/SAM/global attention implementation is modified. Cached query length 1
uses all existing keys; non-square chunks retain their explicit additive mask.
"""

from __future__ import annotations

import types


def _sdpa_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    import torch
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    if output_attentions or self.config.pretraining_tp != 1 or self.training:
        return self._tas_eager_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )
    batch, q_len, _ = hidden_states.shape
    query = (
        self.q_proj(hidden_states)
        .view(batch, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key = (
        self.k_proj(hidden_states)
        .view(batch, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value = (
        self.v_proj(hidden_states)
        .view(batch, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value, seq_len=past_len + q_len)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    if past_key_value is not None:
        key = torch.cat((past_key_value[0], key), dim=2)
        value = torch.cat((past_key_value[1], value), dim=2)
    present = (key, value) if use_cache else None
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    # This shortcut is enabled only by our equal-length, unpadded inference backend.
    causal = False
    if getattr(self, "_tas_unpadded", False):
        if past_len == 0:
            attention_mask, causal = None, q_len > 1
        elif q_len == 1:
            attention_mask = None
    output = F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal,
    )
    output = output.transpose(1, 2).contiguous().reshape(batch, q_len, self.hidden_size)
    return self.o_proj(output), None, present


def set_attention_backend(model, backend: str, *, unpadded: bool = False):
    """Switch a loaded Llama instance without changing weights or allocating a second model."""
    if backend not in {"eager", "sdpa"}:
        raise ValueError(
            "Use eager or sdpa; external FlashAttention compilation is not required."
        )
    backbone = model.get_model() if hasattr(model, "get_model") else model.model
    for layer in backbone.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_tas_eager_forward"):
            if attention.__class__.__name__ != "LlamaAttention":
                raise TypeError(
                    f"Unsupported attention class {attention.__class__.__name__}; pinned eager Llama required."
                )
            attention._tas_eager_forward = attention.forward
        attention._tas_unpadded = unpadded
        attention.forward = (
            types.MethodType(_sdpa_forward, attention)
            if backend == "sdpa"
            else attention._tas_eager_forward
        )
    model._tas_attention_backend = backend
    return backend
