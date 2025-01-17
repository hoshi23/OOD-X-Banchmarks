from dassl.data.datasets import DATASET_REGISTRY

from .fmow_x import FMoWX
from utils.datasets import subsample_classes


@DATASET_REGISTRY.register()
class FMoWIdTestX(FMoWX):
    """Satellite imagery classification.

    62 classes (building or land use categories).
    get id-test and id-val split as test and val split

    Reference:
        - Christie et al. "Functional Map of the World." CVPR 2018.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "fmow/fmow_v1.1"

    def __init__(self, cfg):
        train, val, test = super(FMoWX, self).__init__(cfg)
        name = "fmow"
        val = self.load_additional_split(cfg, name, "id_val")
        test = self.load_additional_split(cfg, name, "id_test")
        labels_all = self.load_classnames()
        labels_list = list(labels_all.keys())
        id_labels: list[int] = None
        ood_labels: list[int] = None
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            id_labels, ood_labels = self.make_random_label_split(labels_list)

        print(f"id_labels: {id_labels}\nood_labels: {ood_labels}")
        train = subsample_classes(train, selected_labels=id_labels)
        test_id = subsample_classes(test, selected_labels=id_labels)
        test_ood = subsample_classes(test, selected_labels=ood_labels)
        super().super_re_init(train_x=train, val=test_id, test=test_ood)
