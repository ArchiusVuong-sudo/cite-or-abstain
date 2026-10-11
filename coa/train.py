"""Local DPO + LoRA run with TRL's DPOTrainer (the trainer Axolotl wraps for `rl: dpo`)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Callable

from . import modeling  # noqa: F401  (sets the loader env before transformers is imported)
from .axolotl_export import export_rows
from .runconfig import RunConfig


def to_trl_rows(rows: list[dict]) -> list[dict]:
    """Axolotl rows -> TRL conversational preference rows."""
    return [{"prompt": r["messages"], "chosen": [r["chosen"]], "rejected": [r["rejected"]]} for r in rows]


def run(conn: sqlite3.Connection, cfg: RunConfig, out_dir: Path, device: str, run_id: int,
        progress: Callable[[int, int, str], None]) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    ex = export_rows(conn)
    if not ex.rows:
        raise ValueError("No active pairs to train on")
    dev = modeling.pick_device(device)
    progress(0, 1, f"Loading {cfg.base_model} on {dev}")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float32)

    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        beta=cfg.beta,
        loss_type=list(cfg.loss_types),
        loss_weights=list(cfg.loss_weights),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.sequence_len,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        bf16=False,
        fp16=False,
        use_cpu=dev == "cpu",
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules="all-linear", task_type="CAUSAL_LM")

    class Live(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):  # noqa: N805
            if not logs or "loss" not in logs:
                return
            conn.execute(
                "INSERT INTO train_metrics VALUES (?,?,?,?,?,?,?)",
                (run_id, state.global_step, logs.get("loss"), logs.get("rewards/margins"),
                 logs.get("rewards/accuracies"), logs.get("learning_rate"), logs.get("epoch")),
            )
            progress(state.global_step, state.max_steps, f"Step {state.global_step} of {state.max_steps}")

    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(to_trl_rows(ex.rows)),
                         processing_class=tok, peft_config=peft_cfg, callbacks=[Live()])
    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    (out_dir / "run_config.json").write_text(json.dumps(cfg.model_dump(), indent=2))
    return {"adapter_dir": str(adapter_dir), "seconds": round(seconds, 1), "steps": trainer.state.global_step,
            "dataset_hash": ex.dataset_hash, "pairs": len(ex.rows), "device": dev}
