from typing import Any, Dict, Tuple, Union

from datasets import Dataset, load_dataset, DatasetDict


def generate_prompt(input1: str, input2: str, response: str, example: str = "") -> str:

    return f"""以下の前提文と仮説文のペアに対して，前提文が仮説文に対してもつ推論関係を、「含意」「矛盾」「中立」のいずれかから答えてください。

{example}### 前提文:
{input1}
### 仮説文:
{input2}
{response}"""


def load_data(shot: int, language: str) -> Tuple[Dataset, str]:
    assert language in {"en", "ja"}
    dsd: DatasetDict = load_dataset("shunk031/JGLUE", name="JNLI")
    map_label: Dict[int, str] = {
        0: "含意",
        1: "矛盾",
        2: "中立",
    }
    shot_example: str = ""
    if shot > 0:
        for i in range(shot):
            dct: Dict[str, Union[str, int]] = dsd["train"].filter(
                lambda ex: ex["label"] == (i % 3)
            )[i // 3]
            shot_example += "### 前提文:\n{}\n### 仮説文:\n{}\n### 返答:\n{}\n\n".format(
                dct["sentence1"], dct["sentence2"], map_label[dct["label"]]
            )
    response: str
    if language == "en":
        response = "### Response:\n"
    elif language == "ja":
        response = "### 返答:\n"
    else:
        raise ValueError(f"Invalid language {language}")

    def process_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompts": generate_prompt(
                example["sentence1"],
                example["sentence2"],
                response=response,
                example=shot_example,
            ),
            "expected": map_label[example["label"]],
        }

    ds = dsd["validation"].map(
        process_dataset,
        remove_columns=[
            "sentence1",
            "label",
            "yjcaptions_id",
            "sentence2",
            "sentence_pair_id",
        ],
    )
    return ds, response
