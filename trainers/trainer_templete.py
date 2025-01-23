import torch


from dassl.engine import TRAINER_REGISTRY


from .trainer_base.ood_trainier_base import OodTrainerX


@TRAINER_REGISTRY.register()
class TrainerTemplete(OodTrainerX):
    """Local regularized Context Optimization (LoCoOp)."""

    def check_cfg(self, cfg):
        """
        Add your custom check_cfg code here
        """
        pass

    def build_model(self):
        """
        Add your custom build_model code here
        """
        pass

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        """
        Add your custom forward_backward code here"""

        loss_summary = {}

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """
        Add your custom load_model code here
        """
        pass

    def output_marshalling(self, output: torch.Tensor) -> torch.Tensor:
        """
        Add your custom output_marshalling code here
        """
        return super().output_marshalling(output=output)

    def ood_score(
        self,
        global_logits: torch.Tensor,
        local_logits: torch.Tensor,
        method: str = "energy",
    ) -> torch.Tensor:
        """
        Add your custom ood_score code here
        """
        return super().ood_score(
            global_logits=global_logits,
            local_logits=local_logits,
            method=method,
        )

    def custom_calcutating_ood_score(
        self,
        output: torch.Tensor,
        method: str = "energy",
    ) -> torch.Tensor:
        """
        Add your custom calcutating_ood_score code here
        """
        return super().custom_calcutating_ood_score(
            output=output,
            method=method,
        )
