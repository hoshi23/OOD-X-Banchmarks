import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


# trainers
import trainers.coop
import trainers.locoop
import trainers.zsclip

# import trainers.idlike
# import trainers.neg_prompt
# import trainers.clipn
# import trainers.neg_label
# import trainers.eoe

# datasets
import datasets.imagenet_x
import datasets.imagenet_r_x
import datasets.imagenet_v2_x
import datasets.imagenet_c_x
import datasets.wilds.iwildcam_x
import datasets.wilds.iwildcam_idtest_x
import datasets.wilds.fmow_x
import datasets.wilds.fmow_idtest_x


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.T:
        cfg.T = args.T

    if args.ood_method:
        cfg.ood_method = args.ood_method

    if args.load_epoch:
        cfg.load_epoch = args.load_epoch

    cfg.eval_ood = args.eval_ood
    cfg.eval_full_supectrum_ood = args.eval_full_supectrum_ood
    cfg.model_dir = args.model_dir


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # Config for LoCoOp and CoOp
    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.LOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.LOCOOP.lambda_value = 0.25

    # # Config for IdLike
    # cfg.TRAINER.ID_LIKE = CN()
    # cfg.TRAINER.ID_LIKE.N_EX_PROMPTS = 100
    # cfg.TRAINER.ID_LIKE.N_CTX = 16
    # cfg.TRAINER.ID_LIKE.CTX_INIT = None
    # cfg.TRAINER.ID_LIKE.CLASS_TOKEN_POSITION = "end"
    # cfg.TRAINER.ID_LIKE.LEARNED_CLS = False
    # cfg.TRAINER.ID_LIKE.N_EX_CTX = 16
    # cfg.TRAINER.ID_LIKE.EX_CTX_INIT = None
    # cfg.TRAINER.ID_LIKE.EX_CLASS_TOKEN_POSITION = "end"
    # cfg.TRAINER.ID_LIKE.EX_LEARNED_CLS = True
    # cfg.TRAINER.ID_LIKE.N_CROP = 256
    # cfg.TRAINER.ID_LIKE.N_SELECTION = 32
    # cfg.TRAINER.ID_LIKE.LAM_IN = 1.0
    # cfg.TRAINER.ID_LIKE.LAM_OUT = 0.3
    # cfg.TRAINER.ID_LIKE.LAM_DIFF = 0.2

    # # Config for NegPrompt
    # cfg.TRAINER.NEGPROMPT = CN()
    # cfg.TRAINER.NEGPROMPT.STAGE = 1
    # cfg.TRAINER.NEGPROMPT.POMP = 0  # True or False?
    # cfg.TRAINER.NEGPROMPT.POMP_k = 128
    # cfg.TRAINER.NEGPROMPT.N_CTX = 16
    # cfg.TRAINER.NEGPROMPT.CTX_INIT = None
    # cfg.TRAINER.NEGPROMPT.CSC = False  # class-specific context
    # cfg.TRAINER.NEGPROMPT.NEGA_CTX = 2
    # cfg.TRAINER.NEGPROMPT.prototype_weight = 0
    # cfg.TRAINER.NEGPROMPT.open_set_method = "OE"  # MSP or Frnce or OE]
    # cfg.TRAINER.NEGPROMPT.open_score = "OE"  # 何に使うか不明
    # cfg.TRAINER.NEGPROMPT.fence_alpha = 0.5
    # cfg.TRAINER.NEGPROMPT.prototype_weight = 0.0
    # cfg.TRAINER.NEGPROMPT.negative_weight = 0.0
    # cfg.TRAINER.NEGPROMPT.distance_weight = 0.0
    # cfg.TRAINER.NEGPROMPT.nega_nega_weight = 0.0
    # cfg.TRAINER.NEGPROMPT.LOSS = "Softmax"

    # # Config for NegLabel
    # cfg.TRAINER.NEGLABEL = CN()
    # cfg.TRAINER.NEGLABEL.ngroup = 100
    # cfg.TRAINER.NEGLABEL.group_fuse_num = None

    # Config for EOE
    # cfg.TRAINER.EOE = CN()
    # cfg.TRAINER.EOE.llm_model = "gpt-3.5-turbo-16k"
    # cfg.TRAINER.EOE.ood_task = "far"  # far', 'near', 'fine_grained'
    # cfg.TRAINER.EOE.L = 500
    # cfg.TRAINER.EOE.generate_class = False
    # cfg.TRAINER.EOE.envisioned_classes_folder = None
    # cfg.TRAINER.EOE.label_split_method = (
    #     "all"  # all or base or random or custom_{customname}
    # )
    # cfg.TRAINER.EOE.ensemble = False
    # cfg.TRAINER.EOE.T = 1
    # cfg.TRAINER.EOE.beta = 0.25
    # cfg.TRAINER.EOE.json_number = 0

    # Config for dataset
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, custom or random
    cfg.DATASET.ID_CLASSES_FILE = ""
    cfg.DATASET.OOD_CLASSES_FILE = ""
    cfg.DATASET.PROMPT = "DEFAULT"  # DEFAULT, or a key in CUSTOM_TEMPLATES


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.build_only:
        print("Build done. Exit.")
        return

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        is_evaled = False
        if args.eval_ood:
            trainer.test_ood(is_fullspectrum=False)
            is_evaled = True
        if args.eval_full_supectrum_ood:
            trainer.test_ood(is_fullspectrum=True)
            is_evaled = True
        if not is_evaled:
            trainer.test()
        if trainer._writer is not None:
            trainer._writer.flush()
            trainer.close_writer()
        return

    if args.model_dir != "":
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
    if not args.no_train:
        trainer.train()
    if trainer._writer is not None:
        trainer._writer.flush()
        trainer.close_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--build-only", action="store_true", help="build trainer only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    parser.add_argument("--T", type=float, default=1, help="temperature parameter")

    parser.add_argument("--ood_method", nargs="+", help="")
    parser.add_argument("--eval_full_supectrum_ood", action="store_true", help="")
    parser.add_argument("--eval_ood", action="store_true", help="")
    args = parser.parse_args()
    main(args)
