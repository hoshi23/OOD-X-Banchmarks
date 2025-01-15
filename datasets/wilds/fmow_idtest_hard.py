from dassl.data.datasets import DATASET_REGISTRY

from .fmow_hard import FMoWHard
from utils.datasets import subsample_classes


@DATASET_REGISTRY.register()
class FMoWIdTestHard(FMoWHard):
    """Satellite imagery classification.

    62 classes (building or land use categories).
    get id-test and id-val split as test and val split

    Reference:
        - Christie et al. "Functional Map of the World." CVPR 2018.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "fmow_v1.1"

    def __init__(self, cfg):
        train, val, test = super(FMoWHard, self).__init__(cfg)
        name = "fmow"
        val = self.load_additional_split(cfg, name, "id_val")
        test = self.load_additional_split(cfg, name, "id_test")
        labels_all = self.load_classnames()
        labels_list = list(labels_all.keys())
        train_subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if train_subsample == "random":
            train_subsample = "base"
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            selected_id, selected_ood = self.make_random_label_split(labels_list)
            print(f"selected_id: {selected_id}\nselected_ood: {selected_ood}")
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
        else:
            train = subsample_classes(
                train,
                labels=labels_list,
                subsample=cfg.DATASET.SUBSAMPLE_CLASSES,
            )
            test_base = subsample_classes(test, labels=labels_list, subsample="base")
            test_new = subsample_classes(test, labels=labels_list, subsample="new")
        super().super_re_init(train_x=train, val=test_base, test=test_new)
