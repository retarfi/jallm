<div id="top"></div>

<h1 align="center"> JaLLM: Japanese Large Language Model</h1>

<p align="center">
  <a href="https://github.com/retarfi/jallm#licenses">
    <img alt="GitHub" src="https://img.shields.io/badge/license-MIT-brightgreen">
  </a>
  <!-- <a href="https://github.com/retarfi/jallm/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/retarfi/jallm.svg">
  </a> -->
</p>

This repository is based on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora).

## Usage
### train.py
train.py is used for tuning.

### fastchat-cli.py
fastchat-cli.py is used for chat in CLI.

```sh
poetry run python fastchat-cli.py \
--model-path <path_to_model_directory> \
(--lora-weight <path_to_lora_directory>) \
(--temperature 0.7) # default: 0
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




## Citation
**There will be another paper. Be sure to check here again when you cite.**

## Licenses
The codes in this repository are distributed under MIT.
