"""
MLX Inference Engine
"""
import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional
import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
import numpy as np

from xomlx.inference.inference_engine import InferenceEngine
from xomlx.download.shard_download import ShardDownloader
from xomlx.inference.shard import Shard
from xomlx.inference.tokenizers import _resolve_tokenizer
from xomlx.helpers import DEBUG, LOG_PATH
from xomlx.inference.mlx_infra.models.general_mha import GeneralMHA
from xomlx.inference.mlx_infra.inference_utils import (
  load_model_config,
  load_model_weights,
  FLOAT_DTYPES,
  check_tied_weights
)

logging.basicConfig( 
	filename=LOG_PATH,
	level=logging.DEBUG,
	format="%(asctime)s - %(levelname)s - %(message)s",
	filemode="a"
)

TEMP = 0.6
TOP_K = 300

class MLXInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.model = None
    self.tokenizer = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      logging.debug("encode called")
      logging.debug(f"shard: {shard}")
      logging.debug(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      return self.tokenizer.encode(
        prompt,
        return_tensors="np"
      )
    
    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )
  
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      logging.debug("decode called")
      logging.debug(f"shard: {shard}")
      logging.debug(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )
  async def _eval_mlx(self, *args):
    await asyncio.get_running_loop().run_in_executor(self.executor, mx.eval, *args)

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"x shape: {x.shape}")
      print(f"x dtype: {x.dtype}")
      print(f"temp: {temp}")
    
    logits = mx.array(x)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)

    sampler = make_sampler(temp=temp, top_p=1.0)
    sample = sampler(logprobs)
    await self._eval_mlx(sample)
    return np.asarray(sample, dtype=int)

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")
      print(f"{input_data.shape=}")

    await self.ensure_shard(shard)
    input_data = mx.array(input_data)

    # check if hidden value
    is_hidden_val = True if getattr(
      input_data, "dtype", None) in FLOAT_DTYPES and getattr(input_data, "ndim", None) == 3 else False

    def infer_wrapper():
      if not is_hidden_val:
        model_out = self.model(input_data)
      else:
        model_out = self.model(input_data, True)
      return np.array(model_out.astype(mx.float32), copy=False), {}

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)
  
  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      logging.debug("shard ensured\n")
      logging.debug(f"shard: {shard}")
      logging.debug(f"class shard: {self.shard}")
      logging.debug(f"uuid: {self.uuid}")

    # reset model after last layer to fix OOM
    if self.shard == shard:
      return

    self.shard = shard
    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    def infer_model():
      logging.debug("start_model called")
      
      check_tied_weights(self.model_path, self.model_config)
      self.model = GeneralMHA(self.model_config, self.shard)
      load_model_weights(self.model_path, self.model, self.model_config)
    
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(infer_model),
    )
        


