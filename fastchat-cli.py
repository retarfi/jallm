"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
import contextlib
import os
from typing import Callable, Optional, Tuple, Union
from unittest.mock import patch

import torch
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle
from fastchat.serve import cli
from fastchat.model.model_adapter import add_model_args, load_model, MPTAdapter
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model, PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from unittest import mock

from jallm.models.model_adapter import PatchedMPTAdapter

MPTAdapter = PatchedMPTAdapter


register_conv_template(
    Conversation(
        name="japanese",
        system="以下はタスクを説明する指示です。要求を適切に満たすような返答を書いてください。\n\n",
        roles=("### 指示", "### 返答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n###",
        stop_str="###",
    )
)


Conversation._get_prompt = Conversation.get_prompt
Conversation._append_message = Conversation.append_message

def conversation_append_message(cls, role: str, message: str):
    cls.offset = -2
    return cls._append_message(role, message)

def conversation_get_prompt_overrider(cls: Conversation) -> str:
    cls.messages = cls.messages[-2:]
    return cls._get_prompt()


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
            print(f"Checkpoint {checkpoint_name} not found")

    if debug:
        print(model)

    return model, tokenizer


@patch("fastchat.serve.inference.load_model", autospec=True)
def main(
    args,
    load_model: Callable,
):
    load_model.return_value = load_lora_model(
        model_path=args.model_path,
        lora_weight=args.lora_weight,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        debug=args.debug,
    )
    print(
        f"model_path: {args.model_path}\n"
        f"lora_weight: {args.lora_weight}\n"
        f"temperature: {args.temperature}\n"
        f"max-new-tokens: {args.max_new_tokens}\n"
        f"device: {args.device}\n"
        f"num_gpus: {args.num_gpus}\n"
        f"max_gpu_memory: {args.max_gpu_memory}\n"
        f"load_8bit: {args.load_8bit}\n"
        f"cpu_offloading: {args.cpu_offloading}\n"
        f"no_context: {args.no_context}\n"
        f"debug: {args.debug}"
    )
    if args.conv_template is not None:
        print(f"conv_template: {args.conv_template}")
        
    with mock.patch.object(Conversation, "get_prompt", conversation_get_prompt_overrider) if args.no_context else contextlib.nullcontext():
        with mock.patch.object(Conversation, "append_message", conversation_append_message) if args.no_context else contextlib.nullcontext():
            cli.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model-path, device, gpus, num-gpus, max-gpu-memory, load-8bit, cpu-offloading
    add_model_args(parser)
    parser.add_argument(
        "--lora-weight",
        type=str,
        help="The path to the lora checkpoint weights",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich"],
        help="Display style.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    main(args)
