import argparse
from typing import Any, Dict

from jallm.training_utils import train, load_params


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # required
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--parameter_file",
        type=str,
        required=True,
        help="json file defining model parameters",
    )
    # optional
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pad_token_id", type=int)
    args = parser.parse_args()

    # parameter configuration
    params: Dict[str, Any] = load_params(
        parameter_file=args.parameter_file, run_name=args.run_name
    )

    params["micro-batch-size"] = args.batch_size
    dummy_name = "search-batch-size"
    train(
        data_path="data/dummy.json",
        output_dir=dummy_name,
        run_name=dummy_name,
        params=params,
        resume_from_checkpoint=False,
        is_japanese=False,
        local_rank=args.local_rank,
        pad_token_id=args.pad_token_id,
        do_only_few_steps=True,
    )
