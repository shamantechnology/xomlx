"""
MLX KVCache implmentation
k,v shape -> (batch_size, num_heads, seq_len, head_dim)
"""
from mlx import nn as mlx_nn
from mlx import core as mlx_core

class KVCache:
  """
  KVCache for storing attention key value pairs
  Using torchtune's KVCache as base
  """
  def __init__(
      self,
      batch_size: int,
      num_heads: int,
      head_dim: int,
      max_seq_len: int
    ):
    cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
    self.k_cache = mlx_core.zeros(cache_shape)
    self.v_cache = mlx_core.zeros(cache_shape)
    self.cache_pos = 0
    self.max_seq_len = max_seq_len

  def size(self):
    return self.k_cache.shape[0]
  
  def reset(self):
    self.k_cache = mlx_core.zeros_like(self.k_cache)
    self.v_cache = mlx_core.zeros_like(self.v_cache)
    self.cache_pos = 0

  def update(self, k_val, v_val):
    bsz, _, seq_len, _ = k_val.shape
    if bsz > self.k_cache.shape[0]:
      raise ValueError("Batch size exceeds cache size")
    if self.cache_pos + seq_len > self.max_seq_len:
      raise ValueError("Model Cache Size Exceeded")

    # possible overflow fix for when reaching cache limit
    # overflow = max(0, self.cache_pos + seq_len - self.max_seq_len)
    # if overflow > 0:
    #   print("!!! Model Cache Size Exceeded !!!")
    #   if overflow < self.cache_pos:
    #     print(f"Shifting cache left by {overflow} positions")
    #     # shift left by 'overflow'
    #     self.k_cache[:, :, :self.cache_pos - overflow, :] = self.k_cache[:, :, overflow:self.cache_pos, :]
    #     self.v_cache[:, :, :self.cache_pos - overflow, :] = self.v_cache[:, :, overflow:self.cache_pos, :]
    #     self.cache_pos -= overflow
    #   else:
    #     print("Dropping entire cache")
    #     # drop everything
    #     self.cache_pos = 0

    self.k_cache[:, :, self.cache_pos:self.cache_pos + seq_len, :] = k_val
    self.v_cache[:, :, self.cache_pos:self.cache_pos + seq_len, :] = v_val
    self.cache_pos += seq_len

    return (
      self.k_cache[:, :, :self.cache_pos, :],
      self.v_cache[:, :, :self.cache_pos, :]
    )