# almost copied from
# https://github.com/tloen/alpaca-lora/blob/630d1146c8b5a968f5bf4f02f50f153a0c9d449d/finetune.py

import argparse
from typing import Any, Dict

import datasets
import psutil
import torch


from jallm.training_utils import load_params, train

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # required
    parser.add_argument(
        "--data_path", type=str, required=True, help="Corpus dataset path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory to save pretrained model",
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--parameter_file",
        type=str,
        required=True,
        help="json file defining model parameters",
    )
    # optional
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--japanese", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pad_token_id", type=int)
    args = parser.parse_args()

    # global variables
    ram_gb: float = psutil.virtual_memory().total / 1073741824
    dev_count: int = max(torch.cuda.device_count(), 1)
    datasets.config.IN_MEMORY_MAX_SIZE = int(ram_gb * 0.9 / dev_count) * 10**9

    # parameter configuration
    params: Dict[str, Any] = load_params(
        parameter_file=args.parameter_file, run_name=args.run_name
    )

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        params=params,
        resume_from_checkpoint=args.resume_from_checkpoint,
        is_japanese=args.japanese,
        local_rank=args.local_rank,
        pad_token_id=args.pad_token_id,
    )
