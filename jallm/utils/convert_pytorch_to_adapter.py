"""
Chat with a model with command line interface.

Usage:
python3 -m jallm.utils.convert_pytorch_to_adapter \
--model ~/model_weights/llama-7b \
--lora-weight ~/model_weights/lora_weight \
--output-dir ~/model_weights/converted_weights
"""
import argparse

from fastchat.model.model_adapter import add_model_args
from peft import PeftModel

from ..modeling_utils import load_lora_model


def main(args):
    model: PeftModel
    model, _ = load_lora_model(
        model_path=args.model_path,
        lora_weight=args.lora_weight,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        debug=False,
    )
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model-path, device, gpus, num-gpus, max-gpu-memory, load-8bit, cpu-offloading
    add_model_args(parser)
    parser.add_argument(
        "--lora-weight",
        type=str,
        help="The path to the lora checkpoint weights",
        required=True,
    )
    parser.add_argument(
        "--output-dir", type=str, help="The output directory of the converted weights"
    )
    args = parser.parse_args()
    main(args)
