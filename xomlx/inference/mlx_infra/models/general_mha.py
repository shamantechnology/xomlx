"""
General MultiHeadAttention for MLX
"""
import math
from typing import Optional, Tuple

from mlx import nn
from mlx import core as mx
from mlx_lm.models.base import create_attention_mask, create_causal_mask

from xomlx.helpers import DEBUG
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
    return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class CachedMultiHeadAttention(nn.Module):
  """
  Cached managed version of MHA
  """
  def __init__(
    self,
    config: dict
  ):
    super().__init__()
    self.config = config
    self.cache = None

    q_out = self.config["num_heads"] * self.config["head_dim"]
    kv_out = self.config["num_kv_heads"] * self.config["head_dim"]
    self.q_proj = nn.Linear(self.config["embed_dim"], q_out, bias=self.config.get("attn_bias", False))
    self.k_proj = nn.Linear(self.config["embed_dim"], kv_out, bias=self.config.get("attn_bias", False))
    self.v_proj = nn.Linear(self.config["embed_dim"], kv_out, bias=self.config.get("attn_bias", False))
    self.o_proj = nn.Linear(q_out, self.config["embed_dim"], bias=self.config.get("attn_bias", False))

    self.rope = nn.RoPE(
      dims=self.config["head_dim"],
      base=self.config["rope_base"],
      scale=self.config.get("rope_scaling_factor", 1.0)
    )

  def setup_cache(self, batch_size: int, max_seq_len: int):
    self.cache = KVCache(
      batch_size=batch_size,
      num_kv_heads=self.config["num_kv_heads"],
      head_dim=self.config["head_dim"],
      max_seq_len=max_seq_len,
      dtype=self.k_proj.weight.dtype
    )

  def __call__(self, x):
    if DEBUG >= 4:
      print("CachedMultiHeadAttention called")
      print(f"x: {x.shape}")

    batch_size, q_len, _ = x.shape
    
    queries = self.q_proj(x)
    keys = self.k_proj(x)
    values = self.v_proj(x)
    
    queries = queries.reshape(
      batch_size, q_len, self.config["num_heads"], self.config["head_dim"]
    ).transpose(0,2,1,3)
    keys = keys.reshape(
      batch_size, q_len, self.config["num_kv_heads"], self.config["head_dim"]
    ).transpose(0,2,1,3)
    values = values.reshape(
      batch_size, q_len, self.config["num_kv_heads"], self.config["head_dim"]
    ).transpose(0,2,1,3)

    if self.cache is not None:
      queries = self.rope(queries, offset=self.cache.cache_pos)
      keys = self.rope(keys, offset=self.cache.cache_pos)
      keys, values = self.cache.update(keys, values)  
    else:
      queries = self.rope(queries)
      keys = self.rope(keys)
      self.setup_cache(batch_size, self.config["max_seq_len"])
      mask = None

    mask = create_attention_mask(x)

    # scaled dot product attention
    scale = self.config["head_dim"] ** -0.5
    attn = mx.fast.scaled_dot_product_attention(
      queries,
      keys,
      values,
      scale=scale,
      mask=mask
    )
    attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, q_len, -1)
    return self.o_proj(attn)

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

  def __call__(self, x):
    y = self.self_attn(self.input_layernorm(x))
    h = x + y
    r = self.mlp(self.post_attention_layernorm(h))
    o = h + r
    return o

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

    if self.config.get("tie_word_embeddings", False):
      self.lm_head = nn.Linear(
        self.config["embed_dim"],
        self.config["vocab_size"],
        bias=False
      )

  def __call__(self, x, is_hidden_val=False):
    curr_layers = [self.layers[i] for i in range(self.shard.start_layer, self.shard.end_layer + 1)]

    if not is_hidden_val:
      h = self.embed_tokens(x)

    for layer in curr_layers:
      h = layer(h)

    if self.shard.end_layer+1 == self.shard.n_layers:
      hn = self.norm(h)

      if self.config.get("tie_word_embeddings", False):
        return self.embed_tokens.as_linear(hn)
      else: 
        return self.lm_head(hn)

    

      
    

