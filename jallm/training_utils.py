# almost copied from
# https://github.com/tloen/alpaca-lora/blob/630d1146c8b5a968f5bf4f02f50f153a0c9d449d/finetune.py

import json
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Tuple, Optional, List

import torch
import transformers
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import PreTrainedModel, PreTrainedTokenizer

from .utils import Prompter


def assert_config(params: Dict[str, Any]) -> None:
    # TODO
    pass


def load_params(parameter_file: str, run_name: str) -> Dict[str, Any]:
    with open(parameter_file, "r") as f:
        params = json.load(f)
    if run_name not in params:
        raise KeyError(f"{run_name} not in parameters.json")
    params = params[run_name]
    assert_config(params)
    return params


def load_model_and_tokenizer(
    base_model: str,
    pad_token_id: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if "llama" in base_model.lower():
        model = transformers.LlamaForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            # device_map=device_map,
        )
        tokenizer = transformers.LlamaTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

    elif "mosaicml/mpt-7b" in base_model:
        from .models.mpt.modeling_mpt import MPTForCausalLM

        config = transformers.AutoConfig.from_pretrained(
            base_model, trust_remote_code=True
        )
        # triton does not work now
        # config.attn_config["attn_impl"] = "triton"

        model = MPTForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b"
        )
        tokenizer.pad_token_id = 1
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        if pad_token_id is not None:
            tokenizer.pad_token_id = pad_token_id
            print(f"pad_token_id is set to {pad_token_id}")
        else:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                print(f"pad_token_id is set to unk_token_id")
            else:
                raise ValueError(
                    "Cannot find pad_token_id. Please specify pad_token_id"
                )
    else:
        if pad_token_id is not None:
            print(
                "pad_token_id is set by user, but tokenizer already has that."
                "So the user's is ignored"
            )

    tokenizer.padding_side = "left"  # Allow batched inference
    return model, tokenizer


def prepare_deepspeed(
    params: Dict[str, Any], per_device_train_batch_size: int, bf16: bool
) -> Optional[Dict[str, Any]]:
    if params["use-deepspeed"]:
        deepspeed = {
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto",
                    "warmup_type": "linear",
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": params["deepspeed-bucket-size"],
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": params["deepspeed-bucket-size"],
                "contiguous_gradients": True,
            },
            # "zero_optimization": {
            #     "stage": 3,
            #     "offload_optimizer": {
            #         "device": "cpu",
            #         "pin_memory": True
            #     },
            #     "offload_param": {
            #         "device": "cpu",
            #         "pin_memory": True
            #     },
            #     "overlap_comm": True,
            #     "contiguous_gradients": True,
            #     "sub_group_size": 1e9,
            #     "reduce_bucket_size": "auto",
            #     "stage3_prefetch_bucket_size": "auto",
            #     "stage3_param_persistence_threshold": "auto",
            #     "stage3_max_live_parameters": 1e9,
            #     "stage3_max_reuse_distance": 1e9,
            #     "stage3_gather_16bit_weights_on_model_save": True
            # },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "steps_per_print": 999999999999,  # logging_steps * gradient_accumulation_steps,
            # "train_batch_size": sum(param_config["batch-size"].values())
            # * gradient_accumulation_steps,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": per_device_train_batch_size,
            "wall_clock_breakdown": False,
        }
        if bf16:
            deepspeed["bf16"] = {"enabled": True}
        else:
            deepspeed["fp16"] = {
                "enabled": "auto",
                # "auto_cast": True, #! Added
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
    else:
        deepspeed = None
    return deepspeed


def generate_and_tokenize_prompt(
    data_point: Dict[str, Any],
    prompter: Prompter,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: Optional[int],
    train_on_inputs: bool,
    truncation: bool = True,
) -> Dict[str, List[int]]:
    def tokenize(prompt: str, add_eos_token=True) -> Dict[str, List[int]]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=truncation,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and (cutoff_len is None or len(result["input_ids"]) < cutoff_len)
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


def train(
    data_path: str,
    output_dir: str,
    run_name: str,
    params: Dict[str, Any],
    is_japanese: bool,
    resume_from_checkpoint: Optional[
        str
    ] = None,  # either training checkpoint or final adapter,
    local_rank: int = -1,
    pad_token_id: Optional[int] = None,
    do_only_few_steps: bool = False,
):
    if "lora" in params.keys():
        print("LoRA is enabled")
        do_lora = True
    else:
        print("LoRA is not enabled")
        do_lora = False
    # model/data params
    base_model: str = params["base-model"]  # the only required argument
    # data_path: str = "yahma/alpaca-cleaned"
    # training hyperparams
    gradient_accumulation_steps = params["gradient-accumulation-steps"]
    per_device_train_batch_size: int = int(
        params["micro-batch-size"] / torch.cuda.device_count()
    )
    num_epochs: int = params["epochs"] if not do_only_few_steps else 0
    learning_rate: float = params["learning-rate"]
    cutoff_len: int = params["cutoff-len"]
    val_set_size: int = 2000 if not do_only_few_steps else 0
    warmup_steps: int = params["warmup-steps"]
    logging_steps: int = params["logging-steps"]
    save_steps: int = params["save-steps"]
    if do_lora:
        # lora hyperparams
        lora_r: int = params["lora"]["r"]
        lora_alpha: int = params["lora"]["alpha"]
        lora_dropout: float = params["lora"]["dropout"]
        lora_target_modules: List[str] = params["lora"]["target-modules"]
    # llm hyperparams
    train_on_inputs: bool = True  # if False, masks out inputs in loss
    group_by_length: bool = False  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_watch: str = ""  # options: false | gradients | all
    wandb_log_model: str = ""  # options: false | true
    prompt_template_name: str
    if is_japanese:
        prompt_template_name = "alpaca_japanese"
    else:
        # The prompt template to use, will default to alpaca.
        prompt_template_name = "alpaca"
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
        if do_lora:
            print(
                f"lora_r: {lora_r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"lora_dropout: {lora_dropout}\n"
                f"lora_target_modules: {lora_target_modules}\n"
            )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template_name)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    bf16: bool = bool("mosaicml/mpt-7b" in base_model)

    deepspeed: Optional[Dict[str, Any]] = prepare_deepspeed(
        params=params,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=bf16,
    )

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Prepare model and tokenizer
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    model, tokenizer = load_model_and_tokenizer(base_model, pad_token_id=pad_token_id)
    if do_lora:
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        lora_config.save_pretrained(output_dir)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name) and do_lora:
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # TODO: confirm the start point is right in Trainer
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name) and do_lora:
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if do_lora:
        # Be more transparent about the % of trainable params.
        model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # TODO: Deprecate if no problem
    # Prepare data
    # def tokenize(prompt, add_eos_token=True):
    #     # there's probably a way to do this with the tokenizer settings
    #     # but again, gotta move fast
    #     result = tokenizer(
    #         prompt,
    #         truncation=True,
    #         max_length=cutoff_len,
    #         padding=False,
    #         return_tensors=None,
    #     )
    #     if (
    #         result["input_ids"][-1] != tokenizer.eos_token_id
    #         and len(result["input_ids"]) < cutoff_len
    #         and add_eos_token
    #     ):
    #         result["input_ids"].append(tokenizer.eos_token_id)
    #         result["attention_mask"].append(1)

    #     result["labels"] = result["input_ids"].copy()

    #     return result

    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = prompter.generate_prompt(
    #         data_point["instruction"],
    #         data_point["input"],
    #         data_point["output"],
    #     )
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     if not train_on_inputs:
    #         user_prompt = prompter.generate_prompt(
    #             data_point["instruction"], data_point["input"]
    #         )
    #         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #         tokenized_full_prompt["labels"] = [
    #             -100
    #         ] * user_prompt_len + tokenized_full_prompt["labels"][
    #             user_prompt_len:
    #         ]  # could be sped up, probably
    #     return tokenized_full_prompt

    data: Dataset
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    train_data: Dataset
    val_data: Optional[Dataset]
    prtl_generate_tokenize: Callable = partial(
        generate_and_tokenize_prompt,
        prompter=prompter,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        train_on_inputs=train_on_inputs,
    )
    if val_set_size > 0:
        train_val: Dataset = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(prtl_generate_tokenize)
        val_data = train_val["test"].shuffle().map(prtl_generate_tokenize)
    else:
        train_data = data["train"].shuffle().map(prtl_generate_tokenize)
        val_data = None

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        # If set to a positive number, the total number of training steps to perform.
        # Overrides num_train_epochs.
        max_steps=-1 if not do_only_few_steps else 20,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        fp16=not bf16,
        bf16=bf16,
        fp16_opt_level="O3",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="epoch" if not do_only_few_steps else "no",
        eval_steps=1000 if val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=10,
        # load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        local_rank=local_rank,
        deepspeed=deepspeed,
        report_to="wandb" if use_wandb and not do_only_few_steps else "none",
        run_name=run_name if use_wandb and not do_only_few_steps else None,
        disable_tqdm=True,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if do_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
