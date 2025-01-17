from dassl.engine.trainer import TrainerX
from dassl.data import DataManager
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from collections import OrderedDict

from utils.evaluation import get_measures
from utils.plot_util import (
    plot_distributio_ood,
)

from .dataset_mapping.full_spectrum import FULL_SPECTRUM_MAPPING


class OodTrainerX(TrainerX):

    @torch.no_grad()
    def test(self, split=None):
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            # NOTE: val_loader is ID data
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            if self.cfg.eval_hard_ood:
                # NOTE: when use hard dataloader, val_loader = ID，test_loader = OOD
                self.test_ood(level="veryhard")
            data_loader = self.test_loader
        ood_results, classification_results = self.test_ood_score(
            data_loader, is_classification_eval=True, is_self_loader=True
        )
        self.report_classification_results(
            classification_results, tag=f"{split}/{self.epoch}/"
        )
        return list(classification_results.values())[0]

    def ood_score(
        self,
        global_logits: torch.Tensor = None,
        local_logits: torch.Tensor = None,
        method: str = "energy",
    ):
        """
        NOTE: calc minus values
        If you define a custom score function, inherit from this.
        """
        global_logits = global_logits / 100.0
        if "mcm" in method:
            smax_global = F.softmax(global_logits / self.cfg.T, dim=-1)
            score = -np.max(smax_global.detach().cpu().numpy(), axis=1)
            if method == "glmcm":
                local_logits_ = local_logits / 100.0
                smax_local = (
                    F.softmax(local_logits_ / self.cfg.T, dim=-1).detach().cpu().numpy()
                )
                local_score = -np.max(smax_local, axis=(1, 2))
                w = 1.0
                score += local_score * w
        elif "energy" in method:
            # energy_T = self.cfg.T
            energy_T = 0.01
            score = (
                -(energy_T * torch.logsumexp(global_logits / energy_T, dim=-1))
                .detach()
                .cpu()
                .numpy()
            )
        elif "max_logit" in method:
            score = -(global_logits.max(-1)[0]).detach().cpu().numpy()
        return score

    def output_marshalling(self, output: torch.Tensor) -> torch.Tensor:
        """
        To examine classification results, format the output.
        If the model output has a unique shape, inherit and use this.
        """
        return output

    def custom_calcutating_ood_score(
        self,
        output: torch.Tensor,
        method: str = "energy",
    ) -> torch.Tensor:
        """
        Format the output and calculate the score for examining OOD scores.
        If the model output shape or score calculation arguments are unique, inherit and use this.
        """
        score = self.ood_score(
            global_logits=output,
            method=method,
        )
        return score

    @torch.no_grad()
    def test_ood_score(
        self,
        dataloader: torch.utils.data.DataLoader,
        is_classification_eval: bool = False,
    ) -> tuple[dict[str, np.ndarray], OrderedDict]:
        self.set_model_mode("eval")
        self.evaluator.reset()
        ood_scores = {ood_method: [] for ood_method in self.cfg.ood_method}
        for batch in tqdm(dataloader):
            inputs, label = self.parse_batch_test(batch)
            output_original = self.model_inference(inputs)
            output = self.output_marshalling(output_original)
            if is_classification_eval:
                self.evaluator.process(output, label)
            for ood_method in self.cfg.ood_method:
                score = self.custom_calcutating_ood_score(
                    output_original, method=ood_method
                )
                if score.ndim == 0:
                    score = score.reshape(1)
                ood_scores[ood_method].append(score)
        classification_results = None
        if is_classification_eval:
            classification_results = self.evaluator.evaluate()

        ood_scores_np = {
            ood_method: np.concatenate(scores)
            for ood_method, scores in ood_scores.items()
        }
        return ood_scores_np, classification_results

    def calc_ood_measures(
        self, id_scores: dict[str, np.ndarray], ood_scores: dict[str, np.ndarray]
    ) -> dict[dict[str, float]]:
        ood_method_list = list(id_scores.keys())
        results = {}
        if ood_method_list != list(ood_scores.keys()):
            raise ValueError("id_scores and ood_scores are not matched")
        for ood_method in ood_method_list:
            auroc, aupr, fpr = get_measures(
                -id_scores[ood_method], -ood_scores[ood_method]
            )
            results[ood_method] = {"AUROC": auroc, "AUPR": aupr, "FPR": fpr}
        return results

    def report_classification_results(
        self, classification_results: OrderedDict, tag: str = ""
    ) -> None:
        for key, value in classification_results.items():
            print_tag = f"{tag}/classification_result/{key}"
            self.write_scalar(print_tag, value, self.epoch)
            print(f"{print_tag}: {value}")

    def build_additional_data_loader(self, dataset_name: str) -> DataManager:
        additional_data_cfg = self.cfg.clone()
        additional_data_cfg.defrost()
        additional_data_cfg.DATASET.NAME = dataset_name
        additional_data_cfg.freeze()
        additional_dm = DataManager(additional_data_cfg)
        return additional_dm

    def test_ood(self, is_fullspectrum: bool = False) -> None:
        """
        Evaluate OOD detection performance.
        if is_fullspectrum is True, evaluate in full-spectrum mode.
        else, evaluate in standard mode.
        """
        mode = "s"
        if is_fullspectrum:
            mode = "fs"
        print(f"Evaluation in full-spectrum mode: {is_fullspectrum}")
        base_dataset_name = self.cfg.DATASET.NAME
        print(f"base dataset: {base_dataset_name}")
        if base_dataset_name not in FULL_SPECTRUM_MAPPING:
            raise ValueError(f"Not supported dataset: {base_dataset_name}")
        try:
            id_datasets = FULL_SPECTRUM_MAPPING[base_dataset_name]["id"][mode]
            ood_datasets = FULL_SPECTRUM_MAPPING[base_dataset_name]["ood"][mode]
        except KeyError:
            raise ValueError(f"Not supported mode: {mode}")

        id_scores_all: dict[str, np.ndarray] = {
            ood_method: np.array([]) for ood_method in self.cfg.ood_method
        }
        ood_scores_all: dict[str, np.ndarray] = {
            ood_method: np.array([]) for ood_method in self.cfg.ood_method
        }
        id_scores_each_dataset: dict[str, dict[str, np.ndarray]] = {
            ood_method: {dataset: np.array([]) for dataset in id_datasets}
            for ood_method in self.cfg.ood_method
        }
        ood_scores_each_dataset: dict[str, dict[str, np.ndarray]] = {
            ood_method: {dataset: np.array([]) for dataset in ood_datasets}
            for ood_method in self.cfg.ood_method
        }

        #  in-distribution
        for i, dataset in enumerate(id_datasets):
            if dataset == base_dataset_name:
                dataloader = self.val_loader
            else:
                additional_dm = self.build_additional_data_loader(dataset)
                dataloader = additional_dm.val_loader
            in_scores, classification_result = self.test_ood_score(
                dataloader, is_classification_eval=True, is_self_loader=True
            )
            self.report_classification_results(classification_result, f"{dataset}-val")
            for ood_method in self.cfg.ood_method:
                id_scores_all[ood_method] = np.concatenate(
                    [id_scores_all[ood_method], in_scores[ood_method]]
                )
                id_scores_each_dataset[dataset] = in_scores

        # out-distribution
        for i, dataset in enumerate(ood_datasets):
            if dataset == base_dataset_name:
                dataloader = self.test_loader
            else:
                additional_dm = self.build_additional_data_loader(dataset)
                dataloader = additional_dm.test_loader
            out_scores, _ = self.test_ood_score(
                dataloader, is_classification_eval=False, is_self_loader=True
            )
            for ood_method in self.cfg.ood_method:
                ood_scores_all[ood_method] = np.concatenate(
                    [ood_scores_all[ood_method], out_scores[ood_method]]
                )
                # report of all patterns
                for id_dataset in id_datasets:
                    measures = self.calc_ood_measures(
                        id_scores_each_dataset[id_dataset], out_scores
                    )
                    print(f"Results for id-{id_dataset}/ood-{dataset}")
                    for measure, value in measures[ood_method].items():
                        tag = f"id-{id_dataset}/ood-{dataset}/{ood_method}/{measure}"
                        print(f"{tag}: {value}")
                        self.write_scalar(tag, value)
                ood_scores_each_dataset[dataset] = out_scores

        # report of all patterns
        measures = self.calc_ood_measures(id_scores_all, ood_scores_all)
        print("Results for all datasets")
        for ood_method, measures in measures.items():
            for measure, value in measures.items():
                tag = f"id-{base_dataset_name}-all/ood-all/{ood_method}/{measure}"
                print(f"{tag}: {value}")
                self.write_scalar(tag, value)
            plot_distributio_ood(
                self.cfg,
                id_scores_all[ood_method],
                ood_scores_all[ood_method],
                "all",
                ood_method,
            )
