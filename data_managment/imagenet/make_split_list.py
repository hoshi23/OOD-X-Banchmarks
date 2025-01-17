import argparse
from pathlib import Path
from typing import Dict, List

from torchvision.datasets import ImageNet


def get_parents_idx_dict(
    imagenet_root: str,
    wordnet_is_a_txt_path: str,
    words_txt_path: str,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    # Get ImageNet wnids
    wnids = ImageNet(imagenet_root, split="val").wnids
    with open(wordnet_is_a_txt_path, "r") as f:
        wn_lines = f.readlines()
    with open(words_txt_path, "r") as f:
        w_lines = f.readlines()

    # make dict[child_wnid: parent_wnid]
    child_to_parent_wnid = {}
    for wn_line in wn_lines:
        parent_wnid, child_wnid = wn_line.split()
        child_to_parent_wnid[child_wnid] = parent_wnid

    # make dict[wnid: name]
    wnid_to_name: Dict[str, str] = {}
    for w_line in w_lines:
        wnid, name = w_line.split("\t")
        wnid_to_name[wnid] = name
    parent_wnid_set = set()

    # make dict[parent_wnid: [child_wnid]]
    for child_wnid in wnids:
        parent_wnid = child_to_parent_wnid[child_wnid]
        parent_wnid_set.add(parent_wnid)
    parent_wnid_list = list(parent_wnid_set)
    parent_wnid_list.sort()
    print("the number of parent classes", len(parent_wnid_list))

    parents_wnid_dict: Dict[str, List[str]] = {}
    parents_name_dict: Dict[str, List[str]] = {}
    parents_idx_dict: Dict[str, List[str]] = {}
    for child_wnid in wnids:
        parent_wnid = child_to_parent_wnid[child_wnid]
        parent_name = wnid_to_name[parent_wnid].rstrip("\n")
        child_name = wnid_to_name[child_wnid].rstrip("\n")
        child_idx = wnids.index(child_wnid)
        if parent_wnid not in parents_wnid_dict:
            parents_wnid_dict[parent_wnid] = []
            parents_name_dict[parent_name] = []
            parents_idx_dict[parent_wnid] = []
        parents_wnid_dict[parent_wnid].append(child_wnid)
        parents_name_dict[parent_name].append(child_name)
        parents_idx_dict[parent_wnid].append(str(child_idx))

    return parents_idx_dict, parents_wnid_dict, parents_name_dict


def make_x_split(
    parents_idx_dict: Dict[str, List[str]],
) -> tuple[List[str], List[str]]:
    is_first_more = False
    first_dataset = []
    second_dataset = []
    for parent_wnid, children_idx_list in parents_idx_dict.items():
        split = len(children_idx_list) // 2
        if len(children_idx_list) % 2 == 1:
            if is_first_more:
                is_first_more = False
            else:
                split += 1
                is_first_more = True
        first_dataset.extend(children_idx_list[:split])
        second_dataset.extend(children_idx_list[split:])
    print(f"len(first_dataset): {len(first_dataset)}")
    print(f"len(second_dataset): {len(second_dataset)}")
    return first_dataset, second_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imagenet_root", help="ImageNet root")
    parser.add_argument("wordnet_is_a_txt_path", help="wordnet.is_a.txt path")
    parser.add_argument("words_txt_path", help="words.txt path")
    parser.add_argument("--split_save_path", "-s", default="s", help="split_save_path")
    args = parser.parse_args()

    imagenet_root: str = args.imagenet_root
    wordnet_is_a_txt_path: str = args.wordnet_is_a_txt_path
    words_txt_path: str = args.words_txt_path

    get_parents_idx_dict, parents_wnid_dict, parents_name_dict = get_parents_idx_dict(
        imagenet_root,
        wordnet_is_a_txt_path,
        words_txt_path,
    )
    first_dataset, second_dataset = make_x_split(get_parents_idx_dict)
    first_dataset_txt = "\n".join(first_dataset)
    second_dataset_txt = "\n".join(second_dataset)
    split_save_path = Path(args.split_save_path)
    split_save_path.mkdir(parents=True, exist_ok=True)
    with open(f"{split_save_path}/first_datasets.txt", "w") as f:
        f.write(first_dataset_txt)
    with open(f"{split_save_path}/second_datasets.txt", "w") as f:
        f.write(second_dataset_txt)
