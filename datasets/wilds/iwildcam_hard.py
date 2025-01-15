import os.path as osp
import pandas as pd
import math
import pickle

from dassl.data.datasets import DATASET_REGISTRY
from .wilds_base_custom import WILDSBaseCustom
from utils.datasets import subsample_classes, make_ramdom_subsample_classes

exclude_labels = ["empty"]


@DATASET_REGISTRY.register()
class IWildCamHard(WILDSBaseCustom):
    """Animal species recognition.

    182 classes (species).

    Reference:
        - Beery et al. "The iwildcam 2021 competition dataset." arXiv 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "iwildcam_v2.0"

    def __init__(self, cfg):
        train, val, test = super().__init__(cfg)
        self.label_to_name_only_exist = self.load_classnames_only_exist()
        labels_all = []
        labels_ood = []
        for k, v in self.label_to_name_only_exist.items():
            if v in exclude_labels:
                labels_ood.append(k)
            else:
                labels_all.append(k)
        self.labels_all = labels_all
        selected_id, selected_ood = self.split_label_custom_ood(
            cfg, labels_all, labels_ood
        )
        print(f"selected_id: {selected_id}\nselected_ood: {selected_ood}")
        train_subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if train_subsample == "random":
            train_subsample = "base"
        train = subsample_classes(
            train,
            labels=selected_id,
            subsample=train_subsample,
            custom=True,
        )
        test_base = subsample_classes(
            test, labels=selected_id, subsample="base", custom=True
        )
        test_new = subsample_classes(
            test, labels=selected_ood, subsample="new", custom=True
        )
        super().super_re_init(train_x=train, val=test_base, test=test_new)

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

    def split_label_custom_ood(
        self, cfg, labels: list[int], ood_labels: list[int]
    ) -> tuple[list[int], list[int]]:
        selected_id = []
        selected_ood = []
        n = len(labels)
        m = math.ceil(n / 2)
        if cfg.DATASET.SUBSAMPLE_CLASSES == "base":
            selected_id = labels[:m]  # take the first half
            selected_ood = labels[m:]  # take the second half
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            preprocessed_classes = osp.join(self.split_fewshot_dir, "split_classes.pkl")
            if osp.exists(preprocessed_classes):
                with open(preprocessed_classes, "rb") as f:
                    data = pickle.load(f)
                    selected_id, selected_ood = (
                        data["base"],
                        data["new"],
                    )
            else:
                selected_id, selected_ood = make_ramdom_subsample_classes(labels=labels)
                data = {"base": selected_id, "new": selected_ood}
                with open(preprocessed_classes, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            selected_id = labels[m:]
            selected_ood = labels[:m]
        selected_ood.extend(ood_labels)
        return selected_id, selected_ood
