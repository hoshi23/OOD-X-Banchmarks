import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from utils.datasets import (
    subsample_classes,
    make_ramdom_subsample_classes,
    read_custom_split_lables_file,
)


@DATASET_REGISTRY.register()
class ImageNetRX(DatasetBase):
    """
    200クラスしかない
    """

    dataset_dir = "imagenet_r"
    base_dataset_dir = "imagenet"  # imagenetのクラスラベルが必要であるため

    def __init__(self, cfg):
        """
        get eval dataset only
        """
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.base_dataset_dir = os.path.join(root, self.base_dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.test_mos_txt = os.path.join(self.dataset_dir, "test_imagenet_r_mos.txt")
        self.test_txt = os.path.join(self.dataset_dir, "test_imagenet_r.txt")

        text_file = os.path.join(self.base_dataset_dir, "classnames.txt")

        classnames = self.read_classnames(text_file)
        test = self.read_data(classnames, "")

        labels = set()
        for item in test:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        id_labels: list[int] = None
        ood_labels: list[int] = None
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            preprocessed_classes = os.path.join(
                self.split_fewshot_dir, "split_random_classes.pkl"
            )
            if os.path.exists(preprocessed_classes):
                with open(preprocessed_classes, "rb") as f:
                    data = pickle.load(f)
                    id_labels, ood_labels = (
                        data["id"],
                        data["ood"],
                    )
            else:
                id_labels, ood_labels = make_ramdom_subsample_classes(labels=labels)
                data = {"id": id_labels, "ood": ood_labels}
                with open(preprocessed_classes, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif cfg.DATASET.SUBSAMPLE_CLASSES == "custom":
            id_labels_file = cfg.DATASET.ID_CLASSES_FILE
            ood_labels_file = cfg.DATASET.OOD_CLASSES_FILE
            id_labels = read_custom_split_lables_file(id_labels_file)
            ood_labels = read_custom_split_lables_file(ood_labels_file)
        print(f"id_labels: {id_labels}\nood_labels: {ood_labels}")
        # NOTE: this train loader is not used. Just for dummy
        train = subsample_classes(test, selected_labels=id_labels)
        test_id = subsample_classes(test, selected_labels=id_labels)
        test_ood = subsample_classes(test, selected_labels=ood_labels)
        super().__init__(train_x=train, val=test_id, test=test_ood)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def get_original_label_index(self, split_dir_name) -> dict[str, int]:
        split_dir = os.path.join(self.base_dataset_dir, f"images/{split_dir_name}")
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        original_label_index = {}
        for label, folder in enumerate(folders):
            original_label_index[folder] = label
        return original_label_index

    def read_data(self, classnames, split_dir_name):
        split_dir = os.path.join(self.image_dir, split_dir_name)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        base_split_dir_name = "val"
        original_label_index = self.get_original_label_index(base_split_dir_name)

        for _, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            label = original_label_index[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
