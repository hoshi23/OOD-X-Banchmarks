import logging  # isort:skip

logging.disable(logging.WARNING)  # isort:skip

import pickle
import logging
import os.path as osp
from wilds import get_dataset as wilds_get_dataset

from dassl.data.datasets import Datum, DatasetBase
from dassl.utils import mkdir_if_missing

# original code: dassl.data.datasets.dg.wilds.wilds_base import WILDSBase


class WILDSBaseCustom(DatasetBase):

    dataset_dir = ""
    relabel_domain = True

    def __init__(self, cfg) -> dict[str, list[Datum]]:
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        name = self.dataset_dir.split("_")[0]
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.preloaded = osp.join(self.dataset_dir, "zhou_preloaded.pkl")
        self.split_fewshot_dir = osp.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        self.label_to_name = self.load_classnames()  # <- ok
        assert isinstance(self.label_to_name, dict)

        if osp.exists(self.preloaded):
            with open(self.preloaded, "rb") as file:
                dataset = pickle.load(file)
                train = dataset["train"]
                val = dataset["val"]
                test = dataset["test"]
        else:
            dataset = wilds_get_dataset(dataset=name, root_dir=root, download=True)
            subset_train = dataset.get_subset("train")
            subset_val = dataset.get_subset("val")
            subset_test = dataset.get_subset("test")

            train = self.read_data(subset_train)
            val = self.read_data(subset_val)
            test = self.read_data(subset_test)

            # Save time for data loading next time
            preloaded = {"train": train, "val": val, "test": test}
            with open(self.preloaded, "wb") as file:
                pickle.dump(preloaded, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Few-shot learning
        k = cfg.DATASET.NUM_SHOTS
        if k > 0:
            seed = cfg.SEED
            preprocessed = osp.join(self.split_fewshot_dir, f"shot_{k}-seed_{seed}.pkl")

            if osp.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                # groups = self.split_dataset_by_domain(train)
                # groups = list(groups.values())
                # groups = self.generate_fewshot_dataset(*groups, num_shots=k)
                # groups = self.generate_fewshot_dataset(train, num_shots=k)
                train = self.generate_fewshot_dataset(train, num_shots=k)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

        return train, val, test

    def super_re_init(self, train_x=None, val=None, test=None):
        super().__init__(train_x=train_x, val=val, test=test)

    def load_classnames(self):
        raise NotImplementedError

    def get_image_path(self, dataset, idx):
        image_name = dataset._input_array[idx]
        image_path = osp.join(self.dataset_dir, image_name)
        return image_path

    def get_label(self, dataset, idx):
        return int(dataset.y_array[idx])

    def get_domain(self, dataset, idx):
        return int(dataset.metadata_array[idx][0])

    def read_data(self, subset):
        items = []
        indices = subset.indices
        dataset = subset.dataset

        for idx in indices:
            image_path = self.get_image_path(dataset, idx)
            label = self.get_label(dataset, idx)
            domain = self.get_domain(dataset, idx)
            classname = self.label_to_name[label]
            item = Datum(
                impath=image_path, label=label, domain=domain, classname=classname
            )
            items.append(item)

        if self.relabel_domain:
            domains = set([item.domain for item in items])
            mapping = {domain: i for i, domain in enumerate(domains)}

            items_new = []

            for item in items:
                item_new = Datum(
                    impath=item.impath,
                    label=item.label,
                    domain=mapping[item.domain],
                    classname=item.classname,
                )
                items_new.append(item_new)

            return items_new

        return items

    def load_additional_split(self, cfg, name: str, split_name: str) -> list[Datum]:
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        additional_preloaded = osp.join(
            self.dataset_dir, f"zhou_preloaded_additional_{split_name}.pkl"
        )
        if osp.exists(additional_preloaded):
            with open(additional_preloaded, "rb") as file:
                dataset = pickle.load(file)
                return dataset[split_name]
        dataset = wilds_get_dataset(dataset=name, root_dir=root, download=True)
        subset = dataset.get_subset(split_name)
        data = self.read_data(subset)
        with open(additional_preloaded, "wb") as file:
            dataset = {split_name: data}
            pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)
        return data
