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
from mlx.utils import tree_flatten
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

)

logging.basicConfig( 
	filename=LOG_PATH,
	level=logging.DEBUG,
	format="%(asctime)s - %(levelname)s - %(message)s",
	filemode="a"
)

TEMP = 0.6

class MLXInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.model = None
    self.tokenizer = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
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
  
  async def sample(self, x: np.ndarray, temp=TEMP) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
    
    x = mx.array(x)

    if temp == 0.0:
      temp = TEMP

    def sample_wrapper():
      sample = mx.random.categorical(x * (1/temp))
      return np.array(sample)

    return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:
    if DEBUG >= 4:
      logging.info("infer_tensor called")
      logging.info(f"shard: {shard}")
      logging.info(f"input_data: {input_data}")
      logging.info(f"{input_data=}")

    await self.ensure_shard(shard)
    input_data = mx.array(input_data)
    print(f"{input_data=}")

    if inference_state is None:
      inference_state = {}

    # check if hidden value
    is_hidden_val = True if getattr(
      input_data, "dtype", None) in FLOAT_DTYPES and getattr(input_data, "ndim", None) == 3 else False

    def infer_wrapper():
      if not is_hidden_val:
        inference_state["mask"] = nn.MultiHeadAttention.create_additive_causal_mask(input_data.shape[1])
        inference_state["mask"] = inference_state["mask"].astype(self.model.embed_tokens.weight.dtype)
        model_out = self.model(input_data, inference_state["mask"])
      else:
        if inference_state is not None:
          model_out = self.model(input_data, inference_state["mask"], True)
        else:
          raise ValueError("mask is missing for hidden values")
      return np.array(model_out), inference_state

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
      
      self.model = GeneralMHA(self.model_config, self.shard)
      print(f"self.model: {self.model}")
      load_model_weights(self.model_path, self.model)
    
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(infer_model),
    )
        


