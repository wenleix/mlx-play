import mlx.core as mx
import mlx.nn as nn
import math

# Attention is all you need
class LlamaAttention(nn.Modlule):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        # TODO: implement RoPE
        self.rope = nn.RoPE(dims // num_heads, traditional=True)

        # Linear is a straightforward x @ W.T and we have implement it in other place,
        #   use the standard one.
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        # Q, K, V first go through linear layers
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extra shapes
        num_heads = self.num_heads
        # B: Batch size; D: hidden dimension * num_heads
        # L: sequence length
        B, L, D = queries.shape


        # Prepare Q, K, V for the attention computation
        # Essesntially did two things:
        # 1. Head dimension and hidden dimension are coalesced in input/output, but need to seperate during computation
        #   (that's what reshape did)
        # 2. Transpose the sequence dimension and the head dimension,
        #   not exactly sure why, but assume to make matmul to
        #   be "lifted" on the right dimension for multi-head (e.g. vmap on the the dimension corresponding to head)
        # (a.k.a multi-head adds another dimension and we need to vmap on it)
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # positional encoding
        # TODO: we should do KV cache here
        queries = self.rope(queries)
        keys = self.rope(keys)

        # Scaled Dot-Product Attention!
        # As mentioned in original paper
        # \[
        #    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + optional_mask) V
        # \]
        scores = (queries * math.sqrt(1 / queries.shape[-1])) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores, axis=-1)

        # OK, after compute scores \matmul values, we need to
        # 1. transpose head dimension with sequence dimension back
        # 2. coalesce head dimension and hidden dimension
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)



