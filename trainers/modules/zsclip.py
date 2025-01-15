import torch

from clip import clip
from clip_w_local import clip as clip_w_local


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())

    return model


def load_clip_w_local_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip_w_local._MODELS[backbone_name]
    model_path = clip_w_local._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip_w_local.build_model(state_dict or model.state_dict())

    return model
