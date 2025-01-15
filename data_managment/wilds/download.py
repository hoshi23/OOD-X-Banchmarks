import os

from wilds import get_dataset as wilds_get_dataset


if __name__ == "__main__":
    save_root_dir = "../../data"

    datasets = [
        # "iwildcam_v2.0",
        "fmow"
    ]
    for dataset in datasets:
        root_dir = os.path.join(save_root_dir, dataset)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        dataset = wilds_get_dataset(dataset=dataset, root_dir=root_dir, download=True)
        print(f"Dataset {dataset.name} downloaded to {root_dir}")
