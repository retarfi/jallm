import logging
import json
import os
import shutil
import urllib
from typing import List

logger = logging.getLogger("__main__").getChild("evaluate.ppl_vqa")

from typing import Dict, Union

from datasets import Dataset, load_dataset


def load_data(data_dir: str = "data/evaluate") -> Dataset:
    jsonl_path: str = os.path.join(data_dir, "vqa.jsonl")
    json_dlpath: str = os.path.join(data_dir, "question_answers.json")
    if not os.path.isfile(jsonl_path):
        os.makedirs(data_dir, exist_ok=True)
        github_url: str = "https://github.com/yahoojapan/ja-vg-vqa/raw/master/question_answers.json.zip"
        logger.info(f"Download from {github_url}")
        zip_filepath: str = os.path.join(data_dir, os.path.basename(github_url))
        data = urllib.request.urlopen(github_url).read()
        with open(zip_filepath, mode="wb") as f:
            f.write(data)
        shutil.unpack_archive(zip_filepath, data_dir)
        os.remove(zip_filepath)
        with open(json_dlpath, "r") as f:
            lst_data: List[
                Dict[str, Union[int, List[Dict[str, Union[str, int, str, List]]]]]
            ] = json.load(f)
        with open(jsonl_path, "w") as f:
            for d in lst_data:
                for d_qa in d["qas"]:
                    json.dump(
                        dict(
                            filter(
                                lambda item: item[0] in {"question", "answer"},
                                d_qa.items(),
                            )
                        ),
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")

        logger.info(f"Data saved at {jsonl_path}")
    ds: Dataset = load_dataset("json", data_files=jsonl_path)["train"]
    ds = ds.rename_columns({"question": "prompts", "answer": "expected"})
    return ds
