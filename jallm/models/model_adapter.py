# refer from fastchat.model.model_adapter.py

from fastchat.conversation import Conversation, get_conv_template
from fastchat.model.model_adapter import BaseAdapter
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

from .mpt.modeling_mpt import MPTForCausalLM


class PatchedMPTAdapter(BaseAdapter):
    """The model adapter for mosaicml/mpt-7b-chat"""

    def match(self, model_path: str):
        return "mpt" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model = MPTForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len=8192,
            **from_pretrained_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("mpt")


class LLaMAdapter(BaseAdapter):
    "Model adapater for vicuna-v1.1"

    def match(self, model_path: str):
        return "llama" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer
