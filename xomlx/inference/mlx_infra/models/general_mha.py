"""
General MultiHeadAttention for MLX
"""
import math
from typing import Optional, Tuple

from mlx import nn
from mlx import core as mx

from xomlx.inference.shard import Shard
from xomlx.inference.mlx_infra.models.kvcache import KVCache

class MultiLayerPerceptron(nn.Module):
  def __init__(
    self,
    dim: int,
    hidden_dim: int,
  ):
    """
    MLP using silu activation
    NOTE: expand for other activations
    """
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, False)
    self.up_proj = nn.Linear(dim, hidden_dim, False)
    self.down_proj = nn.Linear(hidden_dim, dim, False)

  def __call__(self, x):
    a = self.gate_proj(x)
    b = self.up_proj(x)
    x = a * mx.sigmoid(a) * b
    return self.down_proj(x)

class CachedMultiHeadAttention(nn.Module):
  """
  Cached managed version of MHA
  """
  def __init__(
    self,
    config: dict,
    query_input_dims: Optional[int] = None,
    key_input_dims: Optional[int] = None,
    value_input_dims: Optional[int] = None,
    value_dims: Optional[int] = None,
    value_output_dims: Optional[int] = None
  ):
    super().__init__()
    self.config = config
    self.cache = None
    
    if key_input_dims is not None:
      key_input_dims = key_input_dims
    elif self.config["num_kv_heads"] == self.config["num_heads"]:
      key_input_dims = self.config["embed_dim"]
    else:
      key_input_dims = self.config["num_kv_heads"] * self.config["head_dim"]
    
    query_input_dims = query_input_dims or self.config["embed_dim"]
    value_input_dims = value_input_dims or key_input_dims
    value_dims = value_dims or self.config["embed_dim"]
    value_output_dims = value_output_dims or self.config["embed_dim"]

    self.num_heads = self.config["num_heads"]
    self.num_kv_heads = self.config["num_kv_heads"]
    self.q_proj = nn.Linear(
      self.config["embed_dim"],
      query_input_dims,
      bias=config.get("attn_bias", False)
    )
    self.k_proj = nn.Linear(
      self.config["embed_dim"],
      key_input_dims,
      bias=config.get("attn_bias", False)
    )
    self.v_proj = nn.Linear(
      self.config["embed_dim"],
      value_input_dims,
      bias=config.get("attn_bias", False)
    )
    self.o_proj = nn.Linear(
      self.config["embed_dim"],
      value_output_dims,
      bias=config.get("attn_bias", False)
    )

  def setup_cache(self, batch_size: int, max_seq_len: int):
    self.cache = KVCache(
      batch_size=batch_size,
      num_heads=self.config["num_kv_heads"],
      head_dim=self.config["head_dim"],
      max_seq_len=max_seq_len
    )

  def __call__(self, queries, keys, values, mask=None):
    batch_size, q_len, _ = queries.shape

    rope = nn.RoPE(
      dims=self.config["embed_dim"] // self.config["head_dim"],
      base=self.config["rope_base"],
      scale=self.config["rope_scaling_factor"],
      traditional=True
    )
    
    if self.cache is None:
      self.setup_cache(batch_size, self.config["max_seq_len"])
    
    queries = self.q_proj(queries)
    keys = self.k_proj(keys)
    values = self.v_proj(values)

    queries = queries.reshape(
      batch_size, q_len, self.num_heads, -1
    ).transpose(0,2,1,3)
    keys = keys.reshape(
      batch_size, q_len, self.num_kv_heads, -1
    ).transpose(0,2,1,3)
    values = values.reshape(
      batch_size, q_len, self.num_kv_heads, -1
    ).transpose(0,2,1,3)

    if self.cache.cache_pos == 0:
      queries = rope(queries)
      keys = rope(keys)
    else:
      queries = rope(
        queries,
        offset=self.cache.cache_pos
      )
      
      keys = rope(
        keys,
        offset=self.cache.cache_pos
      )

    keys, values = self.cache.update(keys, values)

    if self.num_kv_heads != self.num_heads:
      rep = self.num_heads // self.num_kv_heads
      keys = mx.repeat(keys, rep, axis=1)
      values = mx.repeat(values, rep, axis=1)

    scale = math.sqrt(1 / queries.shape[-1])

    # alignment score calculation
    scaled_queries = (queries * scale)
    scores = scaled_queries @ keys.transpose(0, 1, 3, 2)
    if mask is not None:
      scores += mask
    scores = mx.softmax(scores, axis=-1)
    ctx_out = (scores @ values).transpose(0, 2, 1, 3).reshape(
      batch_size, q_len, -1
    )

    return self.o_proj(ctx_out)

class EncodingLayer(nn.Module):
  def __init__(
    self,
    config: dict
  ):
    super().__init__()
    self.config = config
    self.self_attn =CachedMultiHeadAttention(
      config=self.config
    )

    self.mlp = MultiLayerPerceptron(
      dim=self.config["embed_dim"],
      hidden_dim=self.config["intermediate_dim"]
    )

    self.input_layernorm = nn.RMSNorm(config["embed_dim"], config["norm_eps"])
    self.post_attention_layernorm = nn.RMSNorm(config["embed_dim"], config["norm_eps"])

  def __call__(self, x, mask=None):
    y = self.input_layernorm(x)
    y = self.self_attn(y, y, y, mask=mask)
    x = x + y
    y = self.post_attention_layernorm(x)
    y = self.mlp(y)
    x = x + y
    
    return x

class GeneralMHA(nn.Module):
  def __init__(
    self,
    config: dict,
    shard: Shard
  ):
    super().__init__()
    self.config = config
    self.shard = shard
    self.attn_bias = config.get("attn_bias", False)

    self.embed_tokens = nn.Embedding(
      self.config["vocab_size"],
      self.config["embed_dim"]
    )
    
    # build layers
    self.layers = [None for _ in range(self.shard.n_layers)]
    for i in range(self.shard.start_layer, self.shard.end_layer + 1):
      self.layers[i] = EncodingLayer(self.config)

    self.norm = nn.RMSNorm(
      self.config["embed_dim"],
      self.config["norm_eps"]
    )

    self.output = nn.Linear(
      self.config["embed_dim"],
      self.config["vocab_size"],
      bias=False
    )
  
  def __call__(self, x, mask, is_hidden_val=False):
    curr_layers = [self.layers[i] for i in range(self.shard.start_layer, self.shard.end_layer + 1)]

    if not is_hidden_val:
      x = self.embed_tokens(x)
    
    for layer in curr_layers:
      x = layer(x, mask)

    if self.shard.end_layer+1 == self.shard.n_layers:
      x = self.norm(x)
      return self.output(x[:, -1])
    else:
      return x

    

      
    

