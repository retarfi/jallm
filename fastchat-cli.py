"""
Chat with a model with command line interface.
"""
import argparse
import contextlib
from typing import Callable
from unittest.mock import patch

from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle
from fastchat.serve import cli
from fastchat.model.model_adapter import add_model_args, model_adapters
from unittest import mock

from jallm.models.model_adapter import PatchedMPTAdapter, LLaMAdapter
from jallm.modeling_utils import load_lora_model


for i, adapter in enumerate(model_adapters):
    if adapter.__class__.__name__ == "MPTAdapter":
        model_adapters[i] = PatchedMPTAdapter()
model_adapters.insert(-1, LLaMAdapter())

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
        f"temperature: {args.temperature}\n"
        f"no_context: {args.no_context}\n"
        f"debug: {args.debug}"
    )
    if args.conv_template is not None:
        print(f"conv_template: {args.conv_template}")

    with mock.patch.object(
        Conversation, "get_prompt", conversation_get_prompt_overrider
    ) if args.no_context else contextlib.nullcontext():
        with mock.patch.object(
            Conversation, "append_message", conversation_append_message
        ) if args.no_context else contextlib.nullcontext():
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
