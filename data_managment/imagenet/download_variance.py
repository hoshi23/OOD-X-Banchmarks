import os
import zipfile
import shutil

import gdown

"""
reference from : https://github.com/Jingkang50/OpenOOD/?tab=readme-ov-file
"""


download_id_dict = {
    "benchmark_imglist": "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1",
    "imagenet_v2": "1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho",
    "imagenet_r": "1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU",
    "imagenet_c": "1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt",
}


metadata_dict = {
    "imagenet_v2": [
        "imagenet/test_imagenet_v2_mos.txt",
        "imagenet/test_imagenet_v2.txt",
    ],
    "imagenet_c": [
        "imagenet/test_imagenet_c_mos.txt",
        "imagenet/test_imagenet_c.txt",
    ],
    "imagenet_r": [
        "imagenet/test_imagenet_r_mos.txt",
        "imagenet/test_imagenet_r.txt",
    ],
}


def require_download(path):
    if os.path.exists(path) and os.path.isdir(path):
        if len(os.listdir(path)) > 0:
            print(path + " already exists.")
        return True

    print(filename + " needs download:")
    return False


def download_dataset(dataset: str, save_dir: int):
    store_path = os.path.join(save_dir, dataset, "images")
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    if require_download(store_path):
        print(store_path)
        if not store_path.endswith("/"):
            store_path = store_path + "/"
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + ".zip")
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


if __name__ == "__main__":
    save_root_dir = "../../data"

    datasets = [
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
    ]

    # Download benchmark_imglist of OpenOOD
    if not os.path.exists(os.path.join(save_root_dir, "benchmark_imglist")):
        gdown.download(
            id=download_id_dict["benchmark_imglist"], output=f"{save_root_dir}/"
        )
        file_path = os.path.join(save_root_dir, "benchmark_imglist.zip")
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall(save_root_dir)
        os.remove(file_path)

    # Download datasets
    for dataset in datasets:
        download_dataset(dataset, save_root_dir)
        for meta_file in metadata_dict[dataset]:
            filename = os.path.basename(meta_file)
            if os.path.exists(os.path.join(save_root_dir, dataset, filename)):
                continue
            shutil.move(
                os.path.join(save_root_dir, "benchmark_imglist", meta_file),
                os.path.join(save_root_dir, f"{dataset}/"),
            )

    # Remove non-required imglist data
    shutil.rmtree(os.path.join(save_root_dir, "benchmark_imglist"))
