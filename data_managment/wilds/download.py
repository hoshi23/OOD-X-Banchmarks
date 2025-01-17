import os

from wilds import get_dataset as wilds_get_dataset


if __name__ == "__main__":
    save_root_dir = "../../data"

    datasets = [
        "iwildcam",
        # "fmow"
    ]
    for dataset in datasets:
        root_dir = os.path.join(save_root_dir, dataset)
        dataset = wilds_get_dataset(dataset=dataset, root_dir=root_dir, download=True)
        print(f"Dataset {dataset} downloaded to {root_dir}")
