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
        # L: sequence length (e.g. context length)
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



class LlamaEncoderLayer(nn.Module):
    # I do think "TransformerBlock" is a better name
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        # i guess "y, y, y" is because of the self-attention
        y, cache = self.attention(y, y, y, mask, cache)

        # OK i don't understand how add norm and feed forward works, but that's fine.
        # also i guess https://github.com/ml-explore/mlx-examples/blob/main/llms/llama/llama.py#L120-L122 it's an easier way to read
        x = x + y
        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache

class Llama(nn.Module):
    def __init__(self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = nn.Sequential(*[LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask=mask)
        x = self.norm(x)
        return self.out_proj(x)

