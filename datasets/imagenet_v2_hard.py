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
class ImageNetV2Hard(DatasetBase):
    """
    200クラスしかない
    """

    dataset_dir = "imagenet_v2"
    base_dataset_dir = "imagenet"  # imagenetのクラスラベルが必要であるため

    def __init__(self, cfg):
        """
        get eval dataset only
        """
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.base_dataset_dir = os.path.join(root, self.base_dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.test_mos_txt = os.path.join(self.dataset_dir, "test_imagenet_v2_mos.txt")
        self.test_txt = os.path.join(self.dataset_dir, "test_imagenet_v2.txt")

        text_file = os.path.join(self.base_dataset_dir, "classnames.txt")

        classnames = self.read_classnames(text_file)
        test = self.read_data(classnames, "")

        labels = set()
        for item in test:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            preprocessed_classes = os.path.join(
                self.split_fewshot_dir, "split_classes.pkl"
            )
            if os.path.exists(preprocessed_classes):
                with open(preprocessed_classes, "rb") as f:
                    data = pickle.load(f)
                    original_base_labels, original_new_labels = (
                        data["base"],
                        data["new"],
                    )
            else:
                original_base_labels, original_new_labels = (
                    make_ramdom_subsample_classes(labels=labels)
                )
                data = {"base": original_base_labels, "new": original_new_labels}
                with open(preprocessed_classes, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            base_labels = original_base_labels
            new_labels = original_new_labels
            # for item in train:
            #     if item.classname in original_base_labels:
            #         base_labels.add(item.label)
            #     elif item.classname in original_new_labels:
            #         new_labels.add(item.label)
            # base_labels = list(base_labels)
            # new_labels = list(new_labels)
            print(f"base_labels: {base_labels}\nnew_labels: {new_labels}")
            # NOTE: dummy train dataloader
            train = subsample_classes(
                test, labels=base_labels, subsample="base", custom=True
            )
            test_base = subsample_classes(
                test, labels=base_labels, subsample="base", custom=True
            )
            test_new = subsample_classes(
                test, labels=new_labels, subsample="new", custom=True
            )
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "custom":
            base_labels_file = cfg.DATASET.BASE_CLASSES_FILE
            new_labels_file = cfg.DATASET.NEW_CLASSES_FILE
            base_labels = read_custom_split_lables_file(base_labels_file)
            new_labels = read_custom_split_lables_file(new_labels_file)
            print(f"base_labels: {base_labels}\nnew_labels: {new_labels}")
            # NOTE: dummy train dataloader
            train = subsample_classes(
                test, labels=base_labels, subsample="base", custom=True
            )
            test_base = subsample_classes(
                test, labels=base_labels, subsample="base", custom=True
            )
            test_new = subsample_classes(
                test, labels=new_labels, subsample="new", custom=True
            )
        else:
            # MEMO: cfg.DATASET.SUBSAMPLE_CLASSES=baseにするべき
            # NOTE: dummy train dataloader
            train = subsample_classes(
                test,
                labels=labels,
                subsample=cfg.DATASET.SUBSAMPLE_CLASSES,
            )
            if cfg.DATASET.SUBSAMPLE_CLASSES == "all":
                test_base = subsample_classes(test, labels=labels, subsample="all")
                # NOTE: データ0のもの
                test_new = subsample_classes(test, labels=labels, subsample="all")
            else:
                test_base = subsample_classes(test, labels=labels, subsample="base")
                test_new = subsample_classes(test, labels=labels, subsample="new")
            # super().__init__(train_x=train, test_base=test, test_new=test)
            # super().__init__(train_x=train, test_base=test_base, test_new=test_new)
            # NOTE: データセット変更に伴い、以下のように変更
        super().__init__(train_x=train, val=test_base, test=test_new)

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

    def get_original_label_index(self, split_dir_name) -> dict[int, str]:
        split_dir = os.path.join(self.base_dataset_dir, f"images/{split_dir_name}")
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        original_label_index = {}
        for label, folder in enumerate(folders):
            original_label_index[label] = folder
        return original_label_index

    def read_data(self, classnames, split_dir_name):
        split_dir = os.path.join(self.image_dir, split_dir_name)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        base_split_dir_name = "val"
        original_label_index = self.get_original_label_index(base_split_dir_name)

        for _, folder in enumerate(folders):
            label = int(folder)
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            foldername_original = original_label_index[label]
            classname = classnames[foldername_original]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_data_val(self, classnames):
        with open(os.path.join(self.dataset_dir, "LOC_val_solution.csv"), "r") as f:
            val_annot = [l.strip() for l in f.readlines()]
        val_img2folder = {}
        for annot in val_annot[1:]:
            val_img2folder[annot.split(",")[0]] = annot.split(",")[1].split()[0]
        val_dir = os.path.join(self.image_dir, "val")
        folder2label = {
            folder: i
            for i, folder in enumerate(
                sorted(
                    f.name
                    for f in os.scandir(os.path.join(self.image_dir, "train"))
                    if f.is_dir()
                )
            )
        }
        items = []
        for imname in listdir_nohidden(val_dir):
            impath = os.path.join(val_dir, imname)
            folder = val_img2folder[imname.split(".")[0]]
            classname = classnames[folder]
            label = folder2label[folder]
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)
        return items
