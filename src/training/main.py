import os
import logging
from time import gmtime, strftime, time
from pathlib import Path
import json
import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from clip.clip import _transform, load
from clip.model import convert_weights, CLIP
from clip.model_LORA import convert_weights_LORA, CLIP_LORA
from training.train import train, get_loss, get_loss_multi_label
from training.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr
from mm_retrieval.inf_itm import image_text_ret
from training.test_defs import get_features_feta, calc_nn_feta
from training.utils import check_existing_results
import loralib as lora
import numpy as np
import random



# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return (not args.distributed) or dist.get_rank() == 0 or args.dp


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = int(os.environ.get('RANK', 0))
    setup_worker_logging(args.rank, log_queue, args.log_level)
    if args.distributed:
        dist.init_process_group(backend='nccl')
    # Log and save params.
    if is_master(args):
        if args.debug:
            import pydevd_pycharm
            pydevd_pycharm.settrace(args.debug_ip, port=args.debug_port, stdoutToServer=True, stderrToServer=True,
                                    suspend=False)
            print(f'Debugger set to {args.debug_ip}:{args.debug_port}')
            logging.info(f'Debugger set to {args.debug_ip}:{args.debug_port}')
            args.workers = 0
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    if args.dp:
        args.batch_size *= args.world_size
    if args.lora_r != 0:
        CLIP_class = CLIP_LORA
        convert_weights_func = convert_weights_LORA
    else:
        CLIP_class = CLIP
        convert_weights_func = convert_weights
    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model
    if args.openai_pretrained:
        print("Using openai pretrained model")
        logging.info("Using openai pretrained model")
        model, preprocess_train, preprocess_val = load(args.model, jit=False, is_train=True, args=args)
        # model, preprocess_train, preprocess_val = load(args.model, "cuda", jit=True, is_train=True)
    else:
        model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
        print('Loading model from', model_config_file)
        logging.info('Loading model from', model_config_file)
        assert os.path.exists(model_config_file)
        with open(model_config_file, 'r') as f:
            model_info = json.load(f)
        model = CLIP_class(**model_info, args=args)
        convert_weights_func(model)
        preprocess_train = _transform(model.visual.input_resolution, is_train=True)
        preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights_func(model)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
        if args.precision == "fp16":
            convert_weights_func(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n: not exclude(n)
    if args.lora_r != 0:
        lora.mark_only_lora_as_trainable(model)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        if args.lora_r == 0:
            optimizer_params = [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd}
            ]
        else:
            optimizer_params = [
                {"params": rest_params, "weight_decay": args.wd}
            ]
        optimizer = optim.AdamW(optimizer_params, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = GradScaler() if args.precision == "amp" else None
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.resume == 'auto':
            if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_latest.pt")):
                args.resume = os.path.join(args.checkpoint_path, f"epoch_latest.pt")

        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
            if unexpected_keys:
                raise RuntimeError(f'Encountered unexpected_keys: {unexpected_keys}')
            missing_keys_x_lora = [key for key in missing_keys if 'lora' not in key]
            if missing_keys_x_lora:
                raise RuntimeError(f'Encountered missing_keys not related to LORA: {missing_keys_x_lora}')
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        elif args.resume == 'auto':
            logging.info("=> no checkpoint found at '{}'".format(args.checkpoint_path))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.rank == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    if args.lora_r == 0:
        if args.freeze_net == 'text':
            for param in model.module.transformer.parameters():
                param.requires_grad = False
            print("Text transformer is frozen")
            logging.info("Text transformer is frozen")
        elif args.freeze_net == 'visual':
            for param in model.module.visual.parameters():
                param.requires_grad = False
            print("visual network is frozen")
            logging.info("visual network is frozen")
        elif args.freeze_net == 'all':
            for param in model.module.transformer.parameters():
                param.requires_grad = False
            for param in model.module.visual.parameters():
                param.requires_grad = False
            print("Entire network except text embedding and projection are frozen")
            logging.info("Entire network except text embedding and projection are frozen")
    else:
        lora.mark_only_lora_as_trainable(model)
    calc_nn_func = calc_nn_feta
    if args.choose_one_baseline:
        get_loss_func = get_loss
    else:
        get_loss_func = get_loss_multi_label

    if args.train_data is None:
        get_features_func = get_features_feta
        acc_dict_path = os.path.join(os.path.dirname(args.val_data), f'{args.name}')
        image_text_ret(model=model, eval_dataloader=data['val'].dataloader, get_features=get_features_func,
                       calc_nn_func=calc_nn_func, val_preprocess=preprocess_val, output_path=acc_dict_path,
                       tb_writer=writer, is_master=is_master(args), args=args)
        return
    start_time = time()
    for epoch in range(start_epoch, args.epochs):
        if args.rank == 0:
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, scaler, scheduler, args, writer, loss_func=get_loss_func)
        if args.val_data is not None:
            get_features_func = get_features_feta
            acc_dict_path = os.path.join(os.path.dirname(args.val_data), f'{args.name}_{epoch}')
            if is_master(args):
                image_text_ret(model=model, eval_dataloader=data['val'].dataloader, get_features=get_features_func,
                               calc_nn_func=calc_nn_func, val_preprocess=preprocess_val, output_path=acc_dict_path,
                               tb_writer=writer, is_master=is_master(args), epoch=epoch, args=args)
        # Saving checkpoints.
        if is_master(args):
            torch.save({"epoch": epoch + 1, "name": args.name, "state_dict": get_state_dict(model, args.lora_r),
                        "optimizer": optimizer.state_dict()}, os.path.join(args.checkpoint_path, f"epoch_latest.pt"))
            if (epoch + 1) == args.epochs or (args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0):
                torch.save({"epoch": epoch + 1, "name": args.name, "state_dict": get_state_dict(model, args.lora_r),
                            "optimizer": optimizer.state_dict()},
                           os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"))
        end_time = time()
        print(f'Epoch {epoch} total time: {(end_time - start_time)/60}')
        logging.info(f'Epoch {epoch} total time: {(end_time - start_time)/60}')
        if (args.data_mode == "ikea" and epoch == 5) or \
                (args.data_mode == "cm" and args.exp_mode == 'zero' and epoch == 2):
            # Early stopping IKEA training as most of the gain is already available after 5 epochs
            # Early stopping car manuals zero shot training as most of the gain is already available after 2 epochs
            break
    return 0


def get_state_dict(model_ext, lora_r):
    if lora_r == 0:
        return model_ext.state_dict()
    else:
        return lora.lora_state_dict(model_ext)


def main(args):
    args.gpu = args.local_rank if args.local_rank is not None else args.gpu
    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
            gmtime(),
        )

    if args.copy_codebase:
        import sys
        import subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
            logging.info(f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.")
            return -1
        print(f"Copying codebase to {new_code_path}")
        logging.info(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        logging.info("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        logging.info("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if check_existing_results(args.log_path, args.epochs) and args.train_data:
        print('Full run results already available, cancelling run')
        logging.info('Full run results already available, cancelling run')
        return 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(0)
    assert args.precision in ['amp', 'fp16', 'fp32']
    args.ngpus_per_node = torch.cuda.device_count()

    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    main_worker(args.gpu, None, log_queue, args)
    return 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
