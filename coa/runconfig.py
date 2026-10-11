"""One run config that drives both the Axolotl YAML export and the local TRL run."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    base_model: str
    beta: float = Field(0.1, gt=0, le=1)
    learning_rate: float = Field(5e-5, gt=0, le=1e-2)
    num_epochs: int = Field(2, ge=1, le=10)
    max_steps: int = Field(60, ge=-1, le=2000)  # -1 = run all epochs
    micro_batch_size: int = Field(2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(4, ge=1, le=64)
    sequence_len: int = Field(768, ge=128, le=4096)
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0, le=0.5)
    warmup_steps: int = Field(5, ge=0, le=500)
    lr_scheduler: str = "cosine"
    # DPO sigmoid loss plus an SFT term on the chosen answer (RPO-style): keeps answers short and stopping.
    loss_types: list[str] = Field(default_factory=lambda: ["sigmoid", "sft"])
    loss_weights: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    seed: int = 7

    def axolotl(self, dataset_path: str, output_dir: str, prepared_dir: str) -> dict:
        """Axolotl DPO config. Local RL datasets need `path: <file>` plus `ds_type: json`."""
        return {
            "base_model": self.base_model,
            "rl": "dpo",
            "rl_beta": self.beta,
            "dpo_loss_type": list(self.loss_types),
            "dpo_loss_weights": list(self.loss_weights),
            "chat_template": "tokenizer_default",
            "datasets": [{
                "path": dataset_path,
                "ds_type": "json",
                "split": "train",
                "type": "chat_template.default",
                "field_messages": "messages",
                "field_chosen": "chosen",
                "field_rejected": "rejected",
            }],
            "dataset_prepared_path": prepared_dir,
            "output_dir": output_dir,
            "adapter": "lora",
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_linear": True,
            "sequence_len": self.sequence_len,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps if self.max_steps > 0 else None,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler,
            "warmup_steps": self.warmup_steps,
            "optimizer": "adamw_torch",
            "bf16": False,
            "fp16": False,
            "tf32": False,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "seed": self.seed,
        }

    def mapping(self) -> list[tuple[str, str, str]]:
        """(Axolotl key, TRL / PEFT argument, value) rows for the Train page."""
        return [
            ("base_model", "model", self.base_model),
            ("rl: dpo", "DPOTrainer", "dpo"),
            ("rl_beta", "DPOConfig.beta", str(self.beta)),
            ("dpo_loss_type", "DPOConfig.loss_type", ", ".join(self.loss_types)),
            ("dpo_loss_weights", "DPOConfig.loss_weights", ", ".join(str(w) for w in self.loss_weights)),
            ("adapter: lora", "peft_config=LoraConfig", "lora"),
            ("lora_r", "LoraConfig.r", str(self.lora_r)),
            ("lora_alpha", "LoraConfig.lora_alpha", str(self.lora_alpha)),
            ("lora_dropout", "LoraConfig.lora_dropout", str(self.lora_dropout)),
            ("lora_target_linear: true", "LoraConfig.target_modules", "all-linear"),
            ("sequence_len", "DPOConfig.max_length", str(self.sequence_len)),
            ("micro_batch_size", "per_device_train_batch_size", str(self.micro_batch_size)),
            ("gradient_accumulation_steps", "gradient_accumulation_steps", str(self.gradient_accumulation_steps)),
            ("num_epochs", "num_train_epochs", str(self.num_epochs)),
            ("max_steps", "max_steps", str(self.max_steps)),
            ("learning_rate", "learning_rate", str(self.learning_rate)),
            ("lr_scheduler", "lr_scheduler_type", self.lr_scheduler),
            ("warmup_steps", "warmup_steps", str(self.warmup_steps)),
            ("bf16 / fp16: false", "fp32 weights", "fp32"),
            ("seed", "seed", str(self.seed)),
        ]
