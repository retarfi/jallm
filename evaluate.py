import argparse
import datetime
import os
from typing import Any, Dict, Optional, List, Union

import pandas as pd
import torch
from datasets import Dataset
from fastchat.model.model_adapter import add_model_args
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

import jallm.models.model_adapter
from jallm.evaluate import jnli, ppl_vqa
from jallm.modeling_utils import load_lora_model
from jallm.training_utils import generate_and_tokenize_prompt
from jallm.utils import get_logger, Prompter

BAR_FMT: str = " {n_fmt}/{total_fmt}: {percentage:3.0f}%, [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

logger = get_logger()


def compare_generated(answers: List[str], predicted: List[str]) -> float:
    lst_correct: List[int] = []
    for a, p in zip(answers, predicted):
        if a in p:
            lst_correct.append(1)
        else:
            lst_correct.append(0)
    return sum(lst_correct) / len(lst_correct)


def main(args: argparse.Namespace) -> None:
    is_task_ppl: bool = bool(args.task.split("-")[0] == "ppl")
    model: Union[PreTrainedModel, PeftModel]
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = load_lora_model(
        model_path=args.model_path,
        lora_weight=args.lora_weight,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        debug=False,
    )

    # hyper parameters
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.01,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.00,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token=tokenizer.eos_token_id,
    )
    ds: Dataset
    if args.task == "jnli":
        res_split: str
        ds, res_split = jnli.load_data(shot=args.shot, language=args.format_lang)
        max_new_tokens: int = 5
    elif args.task == "ppl-vqa":
        ds = ppl_vqa.load_data()
    else:
        raise NotImplementedError()
    assert all(
        [x in ds.features.keys() for x in ("prompts", "expected")]
    ), "ds must have 'prompts' and 'expected'"

    if is_task_ppl:
        ds = ds.rename_columns({"prompts": "instruction", "expected": "output"})
        prompt_template_name: str
        if args.format_lang == "ja":
            prompt_template_name = "vqa_japanese"
        elif args.format_lang == "en":
            # The prompt template to use, will default to alpaca.
            prompt_template_name = "vqa"
        else:
            raise NotImplementedError(f"Invalid language: {args.format_lang}")
        prompter = Prompter(prompt_template_name)

        def wrapper_gen_tokenize_prompt(
            example: Dict[str, str]
        ) -> Dict[str, List[int]]:
            example["input"] = ""
            example = generate_and_tokenize_prompt(
                example,
                prompter=prompter,
                tokenizer=tokenizer,
                cutoff_len=None,
                train_on_inputs=False,
                truncation=False,
            )
            assert (
                len(example["input_ids"]) <= args.max_length
            ), f"Exceed max_length {args.max_length} < {len(example['input_ids'])}"
            return example

        ds = ds.map(wrapper_gen_tokenize_prompt)
    else:
        if args.debug:
            logger.info(f"Sample prompt:\n{ds[0]['prompts']}")

        def tokenization(example: Dict[str, Any]) -> Dict[str, Any]:
            example["input_ids"] = tokenizer.encode(example["prompts"])
            assert (
                len(example["input_ids"]) + max_new_tokens <= args.max_length
            ), f"Exceed max_length+max_new_tokens {args.max_length+max_new_tokens} < {len(example['input_ids'])}"
            return example

        lst_expected: List[str] = ds["expected"]
        ds = ds.map(tokenization)
    lst_output: List[Union[str, torch.Tensor]] = []
    device: torch.device = torch.device(args.device)
    with torch.no_grad():
        for i in tqdm(range(len(ds)), bar_format=BAR_FMT):
            input_ids = torch.tensor([ds[i]["input_ids"]]).to(device)
            if is_task_ppl:
                labels: torch.Tensor = torch.tensor([ds[i]["labels"]]).to(device)
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                )
                lst_output.append(output.loss)
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
                s = generation_output.sequences[0].to("cpu")
                output = tokenizer.decode(s, skip_special_tokens=True)
                if args.debug:
                    logger.info(output)
                output = output.split(res_split)[-1].strip()
                if args.debug:
                    logger.info(output)
                lst_output.append(output)

    if is_task_ppl:
        ppl: float = float(torch.exp(torch.stack(lst_output).mean()))
        logger.info(f"perplexity: {ppl:.6f}")
    else:
        acc: float = compare_generated(lst_expected, lst_output)
        logger.info(f"acc: {acc:.4f}")

        # TODO: 選択肢にあるものを回答した確率

        save_dir: str = "materials/evaluate"
        os.makedirs(save_dir, exist_ok=True)
        yyyymmdd_hhmm: str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        file_head: str = f"{args.task}_{yyyymmdd_hhmm}_{args.model_path.split('/')[-1]}"
        if args.lora_weight is not None:
            file_head += "_lora"
        df_content: pd.DataFrame = pd.DataFrame(
            {"ground": lst_expected, "generated": lst_output}
        )
        csv_path: str = os.path.join(save_dir, f"{file_head}.csv")
        df_content.to_csv(csv_path, index=False)
        logger.info(f"Genareted result is output in {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["jnli", "ppl-vqa"],
        required=True,
        help="Task. 'ppl' means perplexity",
    )
    parser.add_argument(
        "--shot", type=int, default=0, help="Number of example given in prompt. Only for tasks without ppl"
    )
    # model-path, device, gpus, num-gpus, max-gpu-memory, load-8bit, cpu-offloading
    add_model_args(parser)
    parser.add_argument(
        "--lora-weight",
        type=str,
        help="The path to the lora checkpoint weights",
    )
    parser.add_argument(
        "--format-lang",
        type=str,
        choices=["ja", "en"],
        help="Conversation prompt language.",
    )
    # parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    # parser.add_argument(
    #     "--style",
    #     type=str,
    #     default="simple",
    #     choices=["simple", "rich"],
    #     help="Display style.",
    # )
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    main(args)
