import os
import json
from pathlib import Path
import re

from mlx import core as mx
FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32, mx.float64}

def load_model_config(model_config_path: Path) -> dict:
  """
  Loads the config.json of the model

  Args:
    model_path (Path): local path to model config json

  Returns:
    dict: The config as a dictionary
  """
  model_config = {}
  with open(model_config_path, encoding="utf-8") as f:
    base_config = json.load(f)

    model_config = {
      "rope_scaling": base_config.get("rope_scaling"),
      "embed_dim": base_config["hidden_size"],
      "num_heads": base_config["num_attention_heads"],
      "head_dim": base_config.get(
        "head_dim",
        base_config["hidden_size"] // base_config["num_attention_heads"],
      ),  # Assuming embed_dim = hidden_size
      "num_kv_heads": base_config["num_key_value_heads"],
      "max_seq_len": base_config["max_position_embeddings"],
      "intermediate_dim": base_config["intermediate_size"],
      "attn_dropout": base_config.get("attention_dropout", 0.0),
      "norm_eps": base_config["rms_norm_eps"],
      "rope_base": base_config["rope_theta"],
      "vocab_size": base_config["vocab_size"],
      "num_layers": base_config["num_hidden_layers"],
      "attn_bias": base_config.get("attention_bias", False),
      "hidden_act": base_config.get("hidden_act", "silu"),
      "torch_dtype": base_config.get("torch_dtype", "bfloat16")
    }

    if model_config.get("rope_scaling", None) is not None:
      model_config["rope_scaling_factor"] = model_config["rope_scaling"].get("rope_factor", 32)

    use_org_seq = bool(os.getenv("TORCH_USE_ORG_SEQ", "False").lower() == "true")
    if use_org_seq and model_config.get("rope_scaling", None) is not None:
      model_config["max_seq_len"] = model_config["rope_scaling"]["original_max_position_embeddings"]

  return model_config

def rewrite_weights(weight_path: str):
    w = mx.load(weight_path)
    remapped = {}
    for k, v in w.items():
        new_k = re.sub(r'^(?:model\.)+', '', k)
        if new_k in remapped and remapped[new_k] is not v:
            raise ValueError(f"Key collision after stripping prefix: '{k}' -> '{new_k}' already exists.")
        remapped[new_k] = v
    if "output.weight" not in remapped and "embed_tokens.weight" in remapped:
        remapped["output.weight"] = remapped["embed_tokens.weight"]
    return remapped

def load_model_weights(
  model_dir: Path,
  model: any
):
  safetensor_files = list(model_dir.glob("*.safetensors"))
  if len(safetensor_files) == 0:
    npz_weights = model_dir / "weights.npz"
    model.load_weights(
      rewrite_weights(str(npz_weights))
    )
  else:
    for safetensor_file in safetensor_files:
      model.load_weights(
        rewrite_weights(str(safetensor_file))
      )
