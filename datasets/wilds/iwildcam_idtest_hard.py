import os.path as osp
import pandas as pd
import pickle

from dassl.data.datasets import DATASET_REGISTRY
from utils.datasets import subsample_classes
from .iwildcam_hard import IWildCamHard

exclude_labels = ["empty"]


@DATASET_REGISTRY.register()
class IWildCamIdTestHard(IWildCamHard):
    """Animal species recognition.

    182 classes (species).
    get id-test and id-val split as test and val split

    Reference:
        - Beery et al. "The iwildcam 2021 competition dataset." arXiv 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "iwildcam_v2.0"

    def __init__(self, cfg):
        train, val, test = super(IWildCamHard, self).__init__(cfg)
        name = "iwildcam"
        val = self.load_additional_split(cfg, name, "id_val")
        test = self.load_additional_split(cfg, name, "id_test")

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
