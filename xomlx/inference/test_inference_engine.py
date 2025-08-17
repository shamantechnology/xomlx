from xomlx.inference.mlx.inference_engine import MLXInferenceEngine
from xomlx.inference.inference_engine import InferenceEngine
from xomlx.download.new_shard_download import NewShardDownloader
from xomlx.inference.shard import Shard
from xomlx.helpers import DEBUG
import os
import asyncio
import numpy as np


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str, n_layers: int):
  prompt = "In a single word only, what is the last name of the current president of the USA?"
  resp_full, _ = await inference_engine_1.infer_prompt("A", shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers), prompt=prompt)
  token_full = await inference_engine_1.sample(resp_full)
  token_full = token_full.reshape(1, -1)
  next_resp_full, _ = await inference_engine_1.infer_tensor(
    "A",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=token_full,
  )

  pp = n_layers // 2
  resp1, _ = await inference_engine_1.infer_prompt("B", shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers), prompt=prompt)
  resp2, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=resp1,
  )
  tokens2 = await inference_engine_1.sample(resp2)
  tokens2 = tokens2.reshape(1, -1)
  resp3, _ = await inference_engine_1.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=0, end_layer=pp, n_layers=n_layers),
    input_data=tokens2,
  )
  resp4, _ = await inference_engine_2.infer_tensor(
    "B",
    shard=Shard(model_id=model_id, start_layer=pp + 1, end_layer=n_layers - 1, n_layers=n_layers),
    input_data=resp3,
  )

  assert np.array_equal(resp_full, resp2)
  assert np.array_equal(next_resp_full, resp4)


asyncio.run(test_inference_engine(MLXInferenceEngine(NewShardDownloader()), MLXInferenceEngine(NewShardDownloader()), "llama-3.2-1b", 16))
