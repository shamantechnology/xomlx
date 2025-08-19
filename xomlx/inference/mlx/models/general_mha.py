"""
General MultiHeadAttention for MLX
"""
import math
from typing import Optional, Tuple
import logging

from mlx import nn
from mlx import core as mx

from xomlx.inference.shard import Shard
from xomlx.helpers import DEBUG, LOG_PATH
from .kvcache import KVCache

logging.basicConfig( 
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32, mx.float64}

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
    
    query_input_dims = query_input_dims or self.config["embed_dim"]
    key_input_dims = key_input_dims or config["embed_dim"]
    value_input_dims = value_input_dims or key_input_dims
    value_dims = value_dims or self.config["embed_dim"]
    value_output_dims = value_output_dims or self.config["embed_dim"]

    self.num_heads = config["num_heads"]
    self.rope = nn.RoPE(
      dim=self.config["head_dim"],
      base=self.config["rope_base"],
      scale=self.config["rope_scale_factor"]
    )
    self.query_proj = nn.Linear(
      query_input_dims,
      self.config["embed_dim"],
      bias=config.get("attn_bias", False)
    )
    self.key_proj = nn.Linear(
      key_input_dims,
      self.config["embed_dim"],
      bias=config.get("attn_bias", False)
    )
    self.value_proj = nn.Linear(
      value_input_dims,
      value_dims,
      bias=config.get("attn_bias", False)
    )
    self.output_proj = nn.Linear(
      value_dims,
      value_output_dims,
      bias=config.get("attn_bias", False)
    )

  def setup_cache(self, batch_size: int, max_seq_len: int):
    self.cache = KVCache(
      batch_size=batch_size,
      num_heads=self.config["num_heads"],
      head_dim=self.config["head_dim"],
      max_seq_len=max_seq_len
    )

  def __call__(self, queries, keys, values, mask=None):
    batch_size, q_len, _ = queries.shape
    
    if self.cache is None:
      self.setup_cache(batch_size, self.config["max_seq_len"])
    
    queries = self.query_proj(queries)
    keys = self.key_proj(keys)
    values = self.value_proj(values)

    queries = queries.reshape(
      batch_size, q_len, self.num_heads, -1
    ).transpose(0,2,1,3)
    keys = keys.reshape(
      batch_size, q_len, self.num_heads, -1
    ).transpose(0,2,1,3)
    values = values.reshape(
      batch_size, q_len, self.num_heads, -1
    ).transpose(0,2,1,3)

    if self.cache.cache_pos == 0:
      queries = self.rope(queries)
      keys = self.rope(keys)
    else:
      queries = self.rope(
        queries,
        offset=self.cache.cache_pos
      )
      
      keys = self.rope(
        keys,
        offset=self.cache.cache_pos
      )

    keys, values = self.cache.update(keys, values)

    scale = math.sqrt(1 / queries.shape[-1])
    scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
    if mask is not None:
      scores += mask
    scores = mx.softmax(scores, axis=-1)
    ctx_out = (scores @ values).transpose(0, 2, 1, 3).reshape(
      batch_size, q_len, -1
    )

    return self.output_proj(ctx_out)

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

    self.sa_norm = nn.RMSNorm(config["embed_dim"], config["norm_eps"])
    self.mlp_norm = nn.RMSNorm(config["embed_dim"], config["norm_eps"])

  def __call__(self, x, mask=None, cache=None):
    y = self.sa_norm(x)
    y = self.self_attn(y, y, y, mask=mask)
    x = x + y
    y = self.mlp_norm(x)
    y = self.mlp(y)
    x = x + y
    
    return x


class GeneralMHA:
  def __init__(
    self,
    config: dict,
    shard: Shard
  ):
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
  
  def __call__(self, x, mask=None):
    # check if hidden value
    is_hidden_val = False if getattr(x, "dtype", None) in FLOAT_DTYPES and getattr(x, "ndim", None) == 3 else True
    curr_layers = [self.layers[i] for i in range(self.shard.start_layer, self.shard.end_layer + 1)]

    if mask is None and not is_hidden_val:
      mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
      mask = mask.astype(self.embed_tokens.weight.dtype)

    if not is_hidden_val:
      x = self.embed_tokens(x)
    
    for layer in curr_layers:
      x = layer(x, mask)

    if self.shard.end_layer+1 == self.shard.n_layers:
      x = self.norm(x)
      return self.output(x)
    else:
      return x

    

      
    

