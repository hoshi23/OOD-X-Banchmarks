import os.path as osp
import pandas as pd
import pickle

from dassl.data.datasets import DATASET_REGISTRY

from .wilds_base_custom import WILDSBaseCustom
from utils.datasets import (
    subsample_classes,
    make_ramdom_subsample_classes,
    read_custom_split_lables_file,
)

CATEGORIES = [
    "airport",
    "airport_hangar",
    "airport_terminal",
    "amusement_park",
    "aquaculture",
    "archaeological_site",
    "barn",
    "border_checkpoint",
    "burial_site",
    "car_dealership",
    "construction_site",
    "crop_field",
    "dam",
    "debris_or_rubble",
    "educational_institution",
    "electric_substation",
    "factory_or_powerplant",
    "fire_station",
    "flooded_road",
    "fountain",
    "gas_station",
    "golf_course",
    "ground_transportation_station",
    "helipad",
    "hospital",
    "impoverished_settlement",
    "interchange",
    "lake_or_pond",
    "lighthouse",
    "military_facility",
    "multi-unit_residential",
    "nuclear_powerplant",
    "office_building",
    "oil_or_gas_facility",
    "park",
    "parking_lot_or_garage",
    "place_of_worship",
    "police_station",
    "port",
    "prison",
    "race_track",
    "railway_bridge",
    "recreational_facility",
    "road_bridge",
    "runway",
    "shipyard",
    "shopping_mall",
    "single-unit_residential",
    "smokestack",
    "solar_farm",
    "space_facility",
    "stadium",
    "storage_tank",
    "surface_mine",
    "swimming_pool",
    "toll_booth",
    "tower",
    "tunnel_opening",
    "waste_disposal",
    "water_treatment_facility",
    "wind_farm",
    "zoo",
]


@DATASET_REGISTRY.register()
class FMoWX(WILDSBaseCustom):
    """Satellite imagery classification.

    62 classes (building or land use categories).

    Reference:
        - Christie et al. "Functional Map of the World." CVPR 2018.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "fmow/fmow_v1.1"

    def __init__(self, cfg):
        train, val, test = super().__init__(cfg)
        labels_all = self.load_classnames()
        labels_list = list(labels_all.keys())
        id_labels: list[int] = None
        ood_labels: list[int] = None

        if cfg.DATASET.SUBSAMPLE_CLASSES == "random":
            id_labels, ood_labels = self.make_random_label_split(labels_list)
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "custom":
            id_labels, ood_labels = self.get_custom_split(cfg)

        print(f"id_labels: {id_labels}\nood_labels: {ood_labels}")
        train = subsample_classes(train, selected_labels=id_labels)
        test_id = subsample_classes(test, selected_labels=id_labels)
        test_ood = subsample_classes(test, selected_labels=ood_labels)
        super().super_re_init(train_x=train, val=test_id, test=test_ood)

    def get_image_path(self, dataset, idx):
        idx = dataset.full_idxs[idx]
        image_name = f"rgb_img_{idx}.png"
        image_path = osp.join(self.dataset_dir, "images", image_name)
        return image_path

    def get_domain(self, dataset, idx):
        # number of regions: 5 or 6
        # number of years: 16
        region_id = int(dataset.metadata_array[idx][0])
        year_id = int(dataset.metadata_array[idx][1])
        return region_id * 16 + year_id

    def load_classnames(self):
        return {i: cat for i, cat in enumerate(CATEGORIES)}

    def load_csv(self):
        csv_path = osp.join(self.dataset_dir, "rgb_metadata.csv")
        df = pd.read_csv(csv_path)
        category_element = df["category"].unique()
        print(
            f"category: {category_element}\nlen: {len(category_element)} <-> {len(CATEGORIES)}"
        )
        return df

    def get_random_split(self, labels: list[int]):
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
