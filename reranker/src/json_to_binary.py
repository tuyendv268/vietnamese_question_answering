import os
import sys
import json
import tqdm
import glob
from argparse import ArgumentParser

from indexed_datasets import IndexedDatasetBuilder


def build_from_json(path_to_jsons, path_to_dest):
    os.makedirs(os.path.dirname(os.path.abspath(path_to_dest)), exist_ok=True)
    builder = IndexedDatasetBuilder(path_to_dest)

    for path_to_json in path_to_jsons:
        sub_paths = glob.glob(path_to_json)
        for path_to_json in sub_paths:
            print(path_to_json)
            try:
                data = json.load(open(path_to_json, encoding="utf-8"))
                assert isinstance(data, list), f"{type(data)}"
                for d in tqdm.tqdm(data):
                    builder.add_item(d)
            except json.decoder.JSONDecodeError:
                import pandas as pd
                # df = pd.read_json(path_to_json, encoding="utf-8", lines=True)
                with open(path_to_json, "r", encoding="utf-8") as f:
                    for d in tqdm.tqdm(f.readlines()):
                        json_obj = json.loads(d.strip())
                        builder.add_item(json_obj)
    builder.finalize()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_dest", type=str)
    parser.add_argument("path_to_jsons", type=str, nargs="+")
    args = parser.parse_args()

    build_from_json(**vars(args))