import pytest
from pydantic import ValidationError

from coa.runconfig import RunConfig


def test_axolotl_config_shape():
    c = RunConfig(base_model="m").axolotl("dpo_train.jsonl", "out", "prepared")
    assert c["rl"] == "dpo" and c["adapter"] == "lora" and c["lora_target_linear"] is True
    ds = c["datasets"][0]
    assert ds["path"] == "dpo_train.jsonl" and ds["ds_type"] == "json" and ds["type"] == "chat_template.default"
    assert c["dpo_loss_type"] == ["sigmoid", "sft"] and c["bf16"] is False and c["sequence_len"] <= 768


def test_mapping_covers_trainer_settings():
    keys = {a for a, _, _ in RunConfig(base_model="m").mapping()}
    assert {"rl_beta", "lora_r", "sequence_len", "micro_batch_size", "learning_rate", "max_steps"} <= keys


def test_runconfig_validates():
    with pytest.raises(ValidationError):
        RunConfig(base_model="m", beta=0)
    with pytest.raises(ValidationError):
        RunConfig(base_model="m", learning_rate=1.0)
