import os.path as osp
import pandas as pd
import math
import pickle

from dassl.data.datasets import DATASET_REGISTRY
from .wilds_base_custom import WILDSBaseCustom
from utils.datasets import (
    subsample_classes,
    make_ramdom_subsample_classes,
    read_custom_split_lables_file,
)

exclude_labels = ["empty"]


@DATASET_REGISTRY.register()
class IWildCamX(WILDSBaseCustom):
    """Animal species recognition.

    182 classes (species).

    Reference:
        - Beery et al. "The iwildcam 2021 competition dataset." arXiv 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "iwildcam/iwildcam_v2.0"

    def __init__(self, cfg):
        train, val, test = super().__init__(cfg)
        self.label_to_name_only_exist = self.load_classnames_only_exist()
        labels_all = []
        labels_exclude = []
        for k, v in self.label_to_name_only_exist.items():
            if v in exclude_labels:
                labels_exclude.append(k)
            else:
                labels_all.append(k)
        self.labels_all = labels_all
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            id_labels, ood_labels = self.get_random_split(labels_all, labels_exclude)
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "custom":
            id_labels, ood_labels = self.get_custom_split(cfg)
        print(f"id_labels: {id_labels}\nood_labels: {ood_labels}")
        train = subsample_classes(train, selected_labels=id_labels)
        test_id = subsample_classes(test, selected_labels=id_labels)
        test_ood = subsample_classes(test, selected_labels=ood_labels)
        super().super_re_init(train_x=train, val=test_id, test=test_ood)

    def get_image_path(self, dataset, idx):
        image_name = dataset._input_array[idx]
        image_path = osp.join(self.dataset_dir, "train", image_name)
        return image_path

    def load_classnames(self):
        df = pd.read_csv(osp.join(self.dataset_dir, "categories.csv"))
        return dict(df["name"])

    def load_classnames_only_exist(self):
        df = pd.read_csv(osp.join(self.dataset_dir, "categories.csv"))
        filterd_df = df[df["y"] != 99999]
        return dict(filterd_df["name"])

    def get_random_split(
        self, labels: list[int], labels_exclude: list[int]
    ) -> tuple[list[int], list[int]]:
        selected_id: list[int] = None
        selected_ood: list[int] = None
        preprocessed_classes = osp.join(
            self.split_fewshot_dir, "split_random_classes.pkl"
        )
        if osp.exists(preprocessed_classes):
            with open(preprocessed_classes, "rb") as f:
                data = pickle.load(f)
                selected_id, selected_ood = (
                    data["id"],
                    data["ood"],
                )
        else:
            selected_id, selected_ood = make_ramdom_subsample_classes(labels=labels)
            data = {"id": selected_id, "ood": selected_ood}
            with open(preprocessed_classes, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        selected_ood.extend(labels_exclude)
        return selected_id, selected_ood

    def get_custom_split(
        self,
        cfg,
    ) -> tuple[list[int], list[int]]:
        id_labels_file = cfg.DATASET.ID_CLASSES_FILE
        ood_labels_file = cfg.DATASET.OOD_CLASSES_FILE
        selected_id = read_custom_split_lables_file(id_labels_file)
        selected_ood = read_custom_split_lables_file(ood_labels_file)
        return selected_id, selected_ood
