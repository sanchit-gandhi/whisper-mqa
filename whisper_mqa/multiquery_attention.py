from typing import Optional, Tuple

import torch
from torch import nn

class WhisperMQAttention(nn.Module):
    """Multi-Query attention from 'Fast Transformer Decoding: One Write-Head is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # Keys and values are shared across heads
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.whisper.modeling_whisper.WhisperAttention._shape
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling  # (bsz, tgt_len, embed_dim)

        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(key_value_states)  # (src_bsz, src_len, head_dim)
            value_states = self.v_proj(key_value_states)  # (src_bsz, src_len, head_dim)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.k_proj(hidden_states)  # (bsz, src_len, head_dim)
            value_states = self.v_proj(hidden_states)  # (bsz, src_len, head_dim)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)  # (bsz, src_len, head_dim)
            value_states = self.v_proj(hidden_states)  # (bsz, src_len, head_dim)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bidirectional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # (bsz, tgt_len, embed_dim) -> (bsz, heads, tgt_len, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        # (bsz, heads, tgt_len, head_dim) -> (batch, heads * tgt_len, head_dim)
        query_states = query_states.reshape(bsz, self.num_heads * tgt_len, self.head_dim)

        # key/value: (bsz, src_len, head_dim)
        src_len = key_states.size(1)

        # (bsz, num_heads * tgt_len, head_dim) x (bsz, head_dim, src_len) -> (bsz, num_heads * tgt_len, src_len)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz, self.num_heads * tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads * tgt_len, src_len)}, but is"
                f" {attn_weights.size()}."
            )

        # -> (bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}."
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}."
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        if output_attentions:
            attn_weights_output = attn_weights
        else:
            attn_weights_output = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # (bsz, num_heads, tgt_len, src_len) -> (bsz, num_heads * tgt_len, src_len)
        attn_probs = attn_probs.view(bsz, self.num_heads * tgt_len, src_len)
        # (bsz, num_heads * tgt_len, src_len) x (bsz, src_len, head_dim) -> (bsz, num_heads * tgt_len, head_dim)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads * tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads * tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}."
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)  # (bsz, tgt_len, embed_dim)

        return attn_output, attn_weights_output, past_key_value