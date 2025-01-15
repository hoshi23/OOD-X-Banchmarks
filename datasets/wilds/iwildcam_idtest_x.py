from dassl.data.datasets import DATASET_REGISTRY
from utils.datasets import subsample_classes
from .iwildcam_x import IWildCamX

exclude_labels = ["empty"]


@DATASET_REGISTRY.register()
class IWildCamIdTestX(IWildCamX):
    """Animal species recognition.

    182 classes (species).
    get id-test and id-val split as test and val split

    Reference:
        - Beery et al. "The iwildcam 2021 competition dataset." arXiv 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "iwildcam_v2.0"

    def __init__(self, cfg):
        train, val, test = super(IWildCamX, self).__init__(cfg)
        name = "iwildcam"
        val = self.load_additional_split(cfg, name, "id_val")
        test = self.load_additional_split(cfg, name, "id_test")

        self.label_to_name_only_exist = self.load_classnames_only_exist()
        labels_all = []
        labels_exclude = []
        for k, v in self.label_to_name_only_exist.items():
            if v in exclude_labels:
                labels_exclude.append(k)
            else:
                labels_all.append(k)
        self.labels_all = labels_all
        id_labels, ood_labels = self.split_label_custom_ood(
            cfg, labels_all, labels_exclude
        )
        print(f"id_labels: {id_labels}\nood_labels: {ood_labels}")
        train = subsample_classes(train, selected_labels=id_labels)
        test_id = subsample_classes(test, selected_labels=id_labels)
        test_ood = subsample_classes(test, selected_labels=ood_labels)
        super().super_re_init(train_x=train, val=test_id, test=test_ood)
