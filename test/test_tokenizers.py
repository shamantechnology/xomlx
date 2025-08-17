import os
import re
from transformers import AutoTokenizer, AutoProcessor
from xomlx.models import model_cards


def test_tokenizer(name, tokenizer, verbose=False):
    print(f"--- {name} ({tokenizer.__class__.__name__}) ---")
    text = "Hello! How can I assist you today? Let me know if you need help with something or just want to chat."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"{encoded=}")
    print(f"{decoded=}")

    reconstructed = ""
    for token in encoded:
      if verbose:
        print(f"{token=}")
        print(f"{tokenizer.decode([token])=}")
      reconstructed += tokenizer.decode([token])
    print(f"{reconstructed=}")

    strip_tokens = lambda s: s.lstrip(tokenizer.decode([tokenizer.bos_token_id])).rstrip(tokenizer.decode([tokenizer.eos_token_id]))
    assert text == strip_tokens(decoded) == strip_tokens(reconstructed)

models = []
for model_id in model_cards:
  for engine_type, repo_id in model_cards[model_id].get("repo", {}).items():
    models.append(repo_id)
models = list(set(models))

verbose = os.environ.get("VERBOSE", "0").lower() == "1"
for m in models:
    test_tokenizer(m, AutoProcessor.from_pretrained(m, use_fast=True, trust_remote_code=True), verbose)
