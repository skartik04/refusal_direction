import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
# HARMTYPES = ['harmless', 'harmful']
HARMTYPES = ['golden', 'control']

# SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')
SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, '{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset


def load_train_dataset(harmtype: str, instructions_only: bool=True, join_weird_comps: bool=False):

    file_path = os.path.join(dataset_dir_path, 'q_a_pairs', f"{harmtype}_train.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset