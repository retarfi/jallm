"""
Very simple cli chat program for
rinna/japanese-gpt-neox-3.6b-instruction-sft.
The model does not inherit old prompts.
"""

import argparse

import torch
from fastchat.serve.inference import load_model
from fastchat.model.model_adapter import add_model_args
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args):
    model_name: str = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
    print(
        f"model_path: {model_name}\n"
        f"temperature: {args.temperature}\n"
        f"max-new-tokens: {args.max_new_tokens}\n"
        f"device: {args.device}\n"
        f"num_gpus: {args.num_gpus}\n"
        f"max_gpu_memory: {args.max_gpu_memory}\n"
        f"load_8bit: {args.load_8bit}\n"
        f"cpu_offloading: {args.cpu_offloading}\n"
        f"debug: {args.debug}"
    )
    model, tokenizer = load_model(
        model_name, args.device, args.num_gpus, args.max_gpu_memory, args.load_8bit, args.cpu_offloading, args.debug
    )

    def chat_loop():
        while True:
            try:
                inp = input("ユーザー: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break

            prompt = f"ユーザー: {inp}<NL>システム: "
            token_ids = tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            with torch.no_grad():
                output_ids = model.generate(
                    token_ids.to(model.device),
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :])
            print(output.replace("<NL>", "\n"))
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": output}, "\n")

    try:
        chat_loop()
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model-path, device, gpus, num-gpus, max-gpu-memory, load-8bit, cpu-offloading
    add_model_args(parser)    
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
