<div id="top"></div>

<h1 align="center"> JaLLM: Japanese Large Language Model</h1>

<p align="center">
  <a href="https://github.com/retarfi/jallm#licenses">
    <img alt="GitHub" src="https://img.shields.io/badge/license-MIT-brightgreen">
  </a>
  <a href="https://github.com/retarfi/jallm/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/retarfi/jallm.svg">
  </a>
</p>

This repository is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora).

## Usage
### Instruct Tuning
instruct_tuning.py is used for tuning.
```sh
WANDB_PROJECT=<PROOJECT_NAME> WANDB_RUN_NAME=<RUN_NAME> \
poetry run python -m torch.distributed.launch --nproc_per_node=2 --node_rank=0 train.py \
--data_path izumi-lab/llm-japanese-dataset \
--output_dir <directory_to_save_checkpoints> \
--run_name llama7b \
--parameter_file params.json \
--local_rank=0
```

### Chat
fastchat-cli.py is used for chat in CLI.

```sh
poetry run python fastchat-cli.py \
--model-path <path_to_model_directory> \
(--lora-weight <path_to_lora_directory>) \
(--temperature 0.7) # default: 0 \
(--conv-template japanese)
```
For LoRA model, adapter_config.json must be in the directory of the model's weight (checkpoint).  
Use `--conv-template dolly_v2` with MPT-7B instruct.  
For more detail about conv-template, please see [FastChat/fastchat/conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)

If you would like to chat with rinna/jaapnese-gpt-neox-3.6b-instruction-sft:
```sh
poetry run python -m jallm.models.rinna_instruct_sft.cli
# detault temperature value is 0.01 (cannot set to 0.0) 
```
Only no-context mode is available (cannot inherit old prompts).

### Convert weights
adapter_config.json must be placed (or copied) in ~/model_weights/lora_weight.
```sh
poetry run python -m jallm.utils.convert_pytorch_to_adapter \
--model ~/model_weights/llama-7b \
--lora-weight ~/model_weights/lora_weight \
--output-dir ~/model_weights/converted_weights
```

### Evaluate model
```sh
poetry run python evaluate.py \
--task {ppl-vqa, jnli} \
--model-path decapoda-research/llama-13b-hf \
--lora-weight lora-weights \
--device cuda \
--format-lang ja \
--max-length 256
```



## Citation
**There will be another paper. Be sure to check here again when you cite.**
```
@preprint{Suzuki2023-llmj,
  title={{日本語インストラクションデータを用いた対話可能な日本語大規模言語モデルのLoRAチューニング}},
  author={鈴木 雅弘 and 平野 正徳 and 坂地 泰紀},
  doi={10.51094/jxiv.422},
  archivePrefix={Jxiv},
  year={2023}
}
```


## Licenses
The codes in this repository are distributed under MIT.
