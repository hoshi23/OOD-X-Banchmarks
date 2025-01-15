import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from utils.datasets import (
    subsample_classes,
    make_ramdom_subsample_classes,
    read_custom_split_lables_file,
)


@DATASET_REGISTRY.register()
class ImageNetHard(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        labels = set()
        for item in train:
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
            train = subsample_classes(
                train, labels=base_labels, subsample="base", custom=True
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
            train = subsample_classes(
                train, labels=base_labels, subsample="base", custom=True
            )
            test_base = subsample_classes(
                test, labels=base_labels, subsample="base", custom=True
            )
            test_new = subsample_classes(
                test, labels=new_labels, subsample="new", custom=True
            )
        else:
            # MEMO: cfg.DATASET.SUBSAMPLE_CLASSES=baseにするべき
            train = subsample_classes(
                train, labels=labels, subsample=cfg.DATASET.SUBSAMPLE_CLASSES
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

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
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
