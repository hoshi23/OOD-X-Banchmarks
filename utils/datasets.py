from dassl.data.datasets import Datum
import math


def subsample_classes(*args, selected_labels: list = None):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        selected_labels (list): what classes to subsample. If None, return all classes.
    """

    if selected_labels is None:
        if len(args) == 1:
            args = args[0]
        return args

    relabeler = {y: y_new for y_new, y in enumerate(selected_labels)}

    output = []
    for dataset in args:
        dataset_new = []
        for item in dataset:
            if item.label not in selected_labels:
                continue
            item_new = Datum(
                impath=item.impath.replace("davidjung", "david"),
                label=relabeler[item.label],
                classname=item.classname,
            )
            dataset_new.append(item_new)
        output.append(dataset_new)
    if len(output) == 1:
        output = output[0]
    return output


def make_ramdom_subsample_classes(
    labels: list,
) -> tuple[list, list]:
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """

    n = len(labels)
    # Divide classes into two halves
    m = math.ceil(n / 2)

    print("make SUBSAMPLE random CLASSES!")
    import random

    selected = random.sample(labels, m)

    other_classes = list(set(labels) - set(selected))

    return selected, other_classes


def read_custom_split_lables_file(filepath: str) -> list[int]:
    """
    split_labels_file: 各lineにクラスindexが存在することを想定
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        classes = [int(line.strip()) for line in lines]
    return classes


def get_harddata_original_name(dataset_name: str) -> str:
    import re

    pattern = r"(\w+)Hard"

    match = re.match(pattern, dataset_name)
    if match:
        return match.group(1)
    else:
        return dataset_name
