"""Model loading and batched generation. Every load in the app goes through here."""

from __future__ import annotations

import os

# Threaded weight loading hangs or crashes on Apple MPS with Transformers 5.x; load serially.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")

from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: str
    name: str


def load(model_id: str, device: str = "auto", adapter_dir: str | None = None) -> LoadedModel:
    dev = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    name = model_id
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        name = f"{model_id} + LoRA"
    model.to(dev)
    model.eval()
    return LoadedModel(model, tok, dev, name)


def render(tok: Any, messages: list[dict[str, str]]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def generate(lm: LoadedModel, conversations: list[list[dict[str, str]]], *, n: int = 1, sample: bool = False,
             temperature: float = 0.8, max_new_tokens: int = 40, batch_size: int = 12, seed: int = 0,
             on_batch=None) -> list[list[str]]:
    """Return n completions per conversation, in order."""
    out: list[list[str]] = []
    tok = lm.tokenizer
    for start in range(0, len(conversations), batch_size):
        chunk = conversations[start:start + batch_size]
        texts = [render(tok, c) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(lm.device)
        torch.manual_seed(seed + start)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id,
                                  "num_return_sequences": n}
        if sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = lm.model.generate(**enc, **kwargs)
        new = gen[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(new, skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append([d.strip() for d in decoded[i * n:(i + 1) * n]])
        if on_batch:
            on_batch(min(start + batch_size, len(conversations)), len(conversations))
    return out
