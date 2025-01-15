import torch

from dassl.engine import TRAINER_REGISTRY

from clip import clip


from trainer_base.ood_trainier_base import OodTrainerX
from .modules.zsclip import (
    load_clip_to_cpu as load_zsclip_to_cpu,
    load_clip_w_local_to_cpu,
)

CUSTOM_TEMPLATES = {
    "DEFAULT": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "MCM": "this is a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(OodTrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_zsclip_to_cpu(cfg)
        clip_model.to(self.device)

        if cfg.DATASET.PROMPT:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.PROMPT]
        else:
            temp = CUSTOM_TEMPLATES["DEFAULT"]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZeroshotCLIPLocal(OodTrainerX):
    """
    Zero-shot learning with CLIP.
    CLIPModel: output with local features.
    if you want to calc gl-mcm, use this.
    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_w_local_to_cpu(cfg)
        clip_model.to(self.device)

        if cfg.DATASET.PROMPT:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.PROMPT]
        else:
            temp = CUSTOM_TEMPLATES["DEFAULT"]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def model_inference(self, image):
        image_features, local_image_features = self.clip_model.encode_image(
            image.type(self.dtype)
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(
            dim=-1, keepdim=True
        )
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        logits_local = logit_scale * local_image_features @ self.text_features.T
        return logits, logits_local

    def output_marshalling(self, output: torch.Tensor) -> torch.Tensor:
        output, local_output = output
        return output

    def custom_calcutating_ood_score(
        self,
        output: torch.Tensor,
        method: str = "energy",
    ) -> torch.Tensor:
        glocal_logits, local_logits = output
        score = self.ood_score(
            global_logits=glocal_logits,
            local_logits=local_logits,
            method=method,
        )
        return score
