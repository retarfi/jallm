import os
from typing import Optional, Tuple, Union

import torch
from fastchat.model.model_adapter import load_model
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model, PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_lora_model(
    model_path: str,
    lora_weight: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: Optional[str] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizerBase]:
    model: Union[PreTrainedModel, PeftModel]
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = load_model(
        model_path=model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        debug=debug,
    )
    if lora_weight is not None:
        # model = PeftModelForCausalLM.from_pretrained(model, model_path, **kwargs)
        config = LoraConfig.from_pretrained(lora_weight)
        model = get_peft_model(model, config)

        # Check the available weights and load them
        checkpoint_name = os.path.join(
            lora_weight, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                lora_weight, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
        # The two files above have a different name depending on how they were saved,
        # but are actually the same.
        if os.path.exists(checkpoint_name):
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            raise ValueError(f"Checkpoint {checkpoint_name} not found")

    if debug:
        print(model)

    return model, tokenizer
