import argparse
import os


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B/32", 'ViT-L/14']:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def get_parser():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--train-data", type=str, default=None, help="Path to pkl file with training data")
    parser.add_argument("--val-data", type=str, default=None, help="Path to pkl file with validation data")
    # Run arguments
    parser.add_argument("--logs", type=str, default="../results/",
                        help="Where to store logs and checkpoints.")
    parser.add_argument("--name", type=str, default=None,
                        help="Optional identifier for the experiment when storing logs. Otherwise use current time.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument("--use-bn-sync", default=False, action="store_true", help="Whether to use batch norm sync.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Specify a single GPU to run the code on for debugging."
                             "Leave at None to use all available GPUs.")
    parser.add_argument("--skip-scheduler", action="store_true", default=False,
                        help="Use this flag to skip the learning rate decay.")
    parser.add_argument("--save-frequency", type=int, default=50, help="How often to save checkpoints.")
    parser.add_argument("--resume", default='auto', type=str,
                        help="Path to latest checkpoint, if auto then searches in logs path for an epoch to"
                             " continue training from (default: auto)")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp", help="Floating point precition.")
    parser.add_argument("--model", choices=["RN50", "RN101", "RN50x4", "ViT-B/32", 'ViT-L/14'], default="RN50",
                        help="Name of the vision backbone to use.")
    parser.add_argument("--openai-pretrained", default=False, action='store_true',
                        help="Use the openai pretrained models.")
    # arguments for distributed training
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:6100", type=str,
                        help="Url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="Distributed backend")
    parser.add_argument("--skip-aggregate", default=True, action="store_false", dest='aggregate',
                        help="Whether to aggregate features across gpus before computing the loss")
    parser.add_argument("--report-to", default='', type=str,
                        help="Options are ['tensorboard']")
    parser.add_argument("--C", type=float, default=3.16, help="Inverse regularizer for logistic reg.")
    parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
    parser.add_argument("--distributed", default=False, action="store_true", help="If true run in distributed mode")
    parser.add_argument("--copy-codebase", default=False, action="store_true",
                        help="If true, we copy the entire base on the log directory, and execute from there.")
    parser.add_argument("--dp", default=False, action="store_true", help="Use DP instead of DDP.")
    parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
                        help="In DP, which GPUs to use for multigpu training")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local GPU ID. Is set automatically when using torch.distributed.launch")
    parser.add_argument("--world_size", default=os.environ.get('WORLD_SIZE', None), type=int,
                        help="The total number of processes running. Is set automatically when using torch.distributed.launch")
    # Method arguments
    parser.add_argument("--mil_max", type=float, default=0, help="Should we use mil max loss? (options:0/1)")
    parser.add_argument("--mil_nce_loss", type=float, default=1, help="Should we use mil_nce loss? (options:0/1)")
    parser.add_argument("--soft_max_mil", type=float, default=0, help="Should we use mil soft_max loss? (options:0/1)")
    parser.add_argument("--save_vis", default=False, action="store_true",
                        help="image_font_path required! Should we save visualizations of the data to tmp_dir? (options:False/ True)")
    parser.add_argument("--tmp_dir", type=str, default="/dccstor/alfassy/tmp",
                        help="Path to save visualisations? only active when save_vis is used.")
    parser.add_argument("--image_font_path", type=str, default="./ocr_utils/arial.ttf",
                        help="Path to Pil Image font, required for save_vis argument")
    parser.add_argument("--all_page_texts", default=False, action="store_true",
                        help="Use all page text without post processing. Used for IKEA by default.")
    parser.add_argument("--text_batch_size", type=int, default=10,
                        help="How many texts should we use in a bag with every image? Only active if "
                             "all_page_texts argument is used.")
    parser.add_argument("--data_mode", type=str, default='default',
                        help="Which dataset is used? options:default/cm, this is needed for train-test fold split."
                             "cm difference is the zero, one, few, many shot experimental settings"
                             "For new data, you can use default or create fold def and add to src/training/data.py line 35")
    parser.add_argument("--exp_mode", type=str, default='many',
                        help="Only used in cm data_mode. which train-test setting should we use? "
                             "options: many/few/one/zero. For new data add support of this param to your fold def"
                             " used in src/training/data.py line 35")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Only used in cm data_mode. Which fold to use for training? In the car-manuals data this "
                             "decides which manufacturer is used for test options:0-4, for the 5 different folds."
                             "Option -1 is to use all data together, it is not used in the paper but can be used during"
                             " training to make sure the training loss goes down. For new data add support for this"
                             " argument to your fold def used in src/training/data.py line 35")
    parser.add_argument("--batch_mode", type=str, default='mix',
                        help="mix - use images and texts from different documents in the same the batch."
                             "sep - only use images and texts from the same document in a batch.")
    parser.add_argument("--seed", type=int, default=0, help="Which random seed to use")
    parser.add_argument("--choose_one_baseline", default=False, action="store_true",
                        help="For each image use 1 random text from the same page as a pair. (options:False/ True)")
    parser.add_argument("--freeze_net", type=str, default='',
                        help="Lock some pretrained weights during training. visual - lock the image model. "
                             "text - lock the text model all- lock both and only leave a small tuning layer.")
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank value, default 0 means do not use LoRA")
    parser.add_argument("--lora_all", default=False, action="store_true",
                        help="Should we use LoRA for all layers? only valid when lora_r!=0")
    parser.add_argument("--lora_lock_text", default=False, action="store_true",
                        help="Should LoRA only train the image model? only valid when lora_r!=0")
    parser.add_argument("--lora_lock_image", default=False, action="store_true",
                        help="Should LoRA only train the text model? only valid when lora_r!=0")
    parser.add_argument("--word_stats", default=False, action="store_true", help="Output data word stats")
    parser.add_argument("--token_stats", default=False, action="store_true", help="Output data token stats")
    parser.add_argument("--debug_ip", type=str, default='', help="IP address to use for debugging with pydevd_pycharm")
    parser.add_argument("--debug_port", type=int, default=55555, help="Port to use for debugging with pydevd_pycharm")
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    # args.aggregate = not args.skip_aggregate
    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
    return args


def parse_args_file(args_list):
    parser = get_parser()

    args = parser.parse_args(args_list)
    # args.aggregate = not args.skip_aggregate
    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
