"""
General MultiHeadAttention for MLX
"""
import math
from typing import Optional, Tuple
import logging

import mlx.nn as mlx_nn
import mlx.core as mlx_core

from xomlx.inference.shard import Shard
from xomlx.helpers import DEBUG, LOG_PATH
from .kvcache import KVCache

logging.basicConfig( 
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

class MultiLayerPerceptron(mlx_nn.Module):
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
    self.gate_proj = mlx_nn.Linear(dim, hidden_dim, False)
    self.down_proj = mlx_nn.Linear(hidden_dim, dim, False)
    self.up_proj = mlx_nn.Linear(dim, hidden_dim, False)

  def __call__(self, x):
    return self.down_proj(
      mlx_nn.silu(self.gate_proj(x)) * self.up_proj(x)
    )

class CachedMultiHeadAttention(mlx_nn.Module):
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
    self.rope = mlx_nn.RoPE(
      dim=self.config["head_dim"],
      base=self.config["rope_base"],
      scale=self.config["rope_scale_factor"]
    )
    self.query_proj = mlx_nn.Linear(
      query_input_dims,
      self.config["embed_dim"],
      bias=config.get("attn_bias", False)
    )
    self.key_proj = mlx_nn.Linear(
      key_input_dims,
      self.config["embed_dim"],
      bias=config.get("attn_bias", False)
    )
    self.value_proj = mlx_nn.Linear(
      value_input_dims,
      value_dims,
      bias=config.get("attn_bias", False)
    )
    self.output_proj = mlx_nn.Linear(
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
    batch_size, max_seq_len, _ = queries.shape
    
    if self.cache is None:
      self.setup_cache(batch_size, max_seq_len)
    
    queries = self.query_proj(queries)
    keys = self.key_proj(keys)
    values = self.value_proj(values)

    queries = queries.reshape(
      batch_size, max_seq_len, self.num_heads, -1
    ).transpose(0,2,1,3)
    keys = keys.reshape(
      batch_size, max_seq_len, self.num_heads, -1
    ).transpose(0,2,1,3)
    values = values.reshape(
      batch_size, max_seq_len, self.num_heads, -1
    ).transpose(0,2,1,3)

    if self.cache.cache_pos == 0:
      queries = self.rope(queries)
      keys = self.rope(keys)
    else:
      queries = self.rope(queries, offset=self.cache.cache_pos)
      keys = self.rope(keys, offset=self.cache.cache_pos)

    keys, values = self.cache.update(keys, values)

    scale = math.sqrt(1 / queries.shape[-1])
    scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
    if mask is not None:
      scores += mask
    scores = mlx_core.softmax(scores, axis=-1)
    ctx_out = (scores @ values).transpose(0, 2, 1, 3).reshape(
      batch_size, max_seq_len, -1
    )

    return self.output_proj(ctx_out)

class EncodingLayer(mlx_nn.Module):
  def __init__(
    self,
    config: dict
  ):
    self.self_attn = mlx_nn.MultiHeadAttention(
      dims=self.config["embed_dim"],
      num_heads=self.config["num_heads"],
      bias=self.attn_bias
    )

    self.mlp = MultiLayerPerceptron(
      dim=self.config["embed_dim"],
      hidden_dim=self.config["intermediate_dim"]
    )

    self.sa_norm = mlx_nn.RMSNorm(config["embed_dim"], config["norm_eps"])
    self.mlp_norm = mlx_nn.RMSNorm(config["embed_dim"], config["norm_eps"])


class GeneralMHA:
  def __init__(
    self,
    config: dict,
    shard: Shard
  ):
    self.config = config
    self.shard = shard
    self.attn_bias = config.get("attn_bias", False)
    self.rope = mlx_nn.RoPE(
      dim=self.config["head_dim"],
      base=self.config["rope_base"],
      scale=self.config["rope_scale_factor"]
    )
    
    # build layers
    self.layers = [None for _ in range(self.shard.n_layers)]
    for i in range(self.shard.start_layer, self.shard.end_layer + 1):
      layer = EncodingLayer(config)
      self.layers[i] = layer

    

      
    

