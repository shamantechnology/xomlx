"""
MLX KVCache implmentation
k,v shape -> (batch_size, num_kv_heads, seq_len, head_dim)
"""
from mlx import nn as mlx_nn
from mlx import core as mx

class KVCache:
  """
  KVCache for storing attention key value pairs
  Using torchtune's KVCache as base
  """
  def __init__(
      self,
      batch_size: int,
      num_kv_heads: int,
      head_dim: int,
      max_seq_len: int,
      dtype: mx.Dtype
    ):
    self.batch_size = batch_size
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.cache_pos = 0
    self.max_seq_len = max_seq_len
    self.step = 256

    cache_shape = (self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
    self.k_cache = mx.zeros(cache_shape, dtype=dtype)
    self.v_cache = mx.zeros(cache_shape, dtype=dtype)

  def size(self):
    return self.k_cache.shape[0]
  
  def reset(self):
    self.k_cache = mx.zeros_like(self.k_cache)
    self.v_cache = mx.zeros_like(self.v_cache)
    self.cache_pos = 0

  def update(self, k_vals, v_vals):
    bsz, n_kv_heads, seq_len, k_hdim = k_vals.shape
    assert bsz == self.batch_size and n_kv_heads == self.num_kv_heads and k_hdim == self.head_dim
    prev = self.cache_pos
    if self.cache_pos + seq_len > self.max_seq_len:
      v_hdim = v_vals.shape[3]
      n_steps = (self.step + k_vals.shape[2] - 1) // self.step
      k_shape = (bsz, n_kv_heads, n_steps * self.step, k_hdim)
      v_shape = (bsz, n_kv_heads, n_steps * self.step, v_hdim)
      new_k = mx.zeros(k_shape, k_vals.dtype)
      new_v = mx.zeros(v_shape, v_vals.dtype)
      if self.k_cache is not None:
        if prev % self.step != 0:
          self.k_cache = self.k_cache[..., :prev, :]
          self.v_cache = self.v_cache[..., :prev, :]
        self.k_cache = mx.concatenate([self.k_cache, new_k], axis=2)
        self.v_cache = mx.concatenate([self.v_cache, new_v], axis=2)
      else:
        self.k_cache, self.v_cache = new_k, new_v
    
    self.cache_pos += seq_len
    self.k_cache[:, :, prev:self.cache_pos, :] = k_vals
    self.v_cache[:, :, prev:self.cache_pos, :] = v_vals
    
    return (
      self.k_cache[:, :, :self.cache_pos, :],
      self.v_cache[:, :, :self.cache_pos, :]
    )