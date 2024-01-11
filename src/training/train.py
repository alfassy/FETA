import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.functional import cross_entropy
import logging


def is_master(args):
    return (not args.distributed) or args.rank == 0


def get_loss(model, images, texts, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    tb_loss = {'images': {}}
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features, device=image_features.device) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features, device=image_features.device) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
    tb_loss['images']['i2t_dist'] = torch.nn.functional.softmax(logits_per_image, dim=1)
    tb_loss['images']['t2i_dist'] = torch.nn.functional.softmax(logits_per_image.t(), dim=1)
    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    return total_loss, tb_loss


def get_loss_multi_label(model, images, texts, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    tb_loss = {'images': {}}
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features, device=image_features.device) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features, device=image_features.device) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1:]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()


    total_loss = None
    tb_loss['images']['i2t_dist'] = torch.nn.functional.softmax(logits_per_image, dim=1)
    tb_loss['images']['t2i_dist'] = torch.nn.functional.softmax(logits_per_image.t(), dim=1)
    if args.all_page_texts:
        mil_num = args.text_batch_size
    else:
        mil_num = 5
    if args.mil_max != 0:
        images_num = logits_per_image.shape[0]  # N
        eye = torch.eye(images_num, dtype=torch.bool).unsqueeze(-1).repeat(1, 1, mil_num)
        # move the mil_num different texts per image to their own dimension
        i_t_5 = logits_per_image.reshape((images_num, images_num, mil_num))  # N x N x K
        pos, pos_idx = i_t_5[eye].reshape(images_num, mil_num).max(dim=1)
        neg_mat = i_t_5[~eye].reshape(images_num, (images_num-1)*mil_num)
        full_mat = torch.cat((pos.unsqueeze(-1), neg_mat), dim=1)
        gt_image_text = torch.zeros(images_num, dtype=int)
        if args.gpu is not None:
            gt_image_text = gt_image_text.cuda(args.gpu, non_blocking=True)
        img_text_loss = loss_img(full_mat, gt_image_text)
        # image to text
        txt_logits = i_t_5[:, torch.arange(images_num), pos_idx].t()
        ground_truth_text_image = torch.arange(logits_per_text.shape[1]).long()
        if args.gpu is not None:
            ground_truth_text_image = ground_truth_text_image.cuda(args.gpu, non_blocking=True)
        total_loss = args.mil_max * (img_text_loss +
                                       loss_txt(txt_logits, ground_truth_text_image)) / 2
    if args.mil_nce_loss != 0.:
        x = logits_per_image
        images_num = logits_per_image.shape[0]
        x = x.view(images_num, images_num, -1)
        nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        total_loss = torch.mean(denominator - nominator)
        return total_loss, tb_loss
    if args.soft_max_mil != 0.:
        try:
            softmax_scale = model.softmax_scale.exp().mean()
        except:
            softmax_scale = model.module.softmax_scale.exp().mean()
        images_num = logits_per_image.shape[0]  # N
        eye = torch.eye(images_num, dtype=torch.bool).unsqueeze(-1).repeat(1, 1, mil_num)
        # move the mil_num different texts per image to their own dimension
        i_t_5 = logits_per_image.reshape((images_num, images_num, mil_num))  # N x N x K
        all_pos = i_t_5[eye].reshape(images_num, mil_num)
        _, pos_idx = all_pos.max(dim=1)
        pos = (torch.nn.functional.softmax(softmax_scale*all_pos.detach(),dim=1)* all_pos).sum(dim=1)
        neg_mat = i_t_5[~eye].reshape(images_num, (images_num - 1) * mil_num)
        full_mat = torch.cat((pos.unsqueeze(-1), neg_mat), dim=1)
        gt_image_text = torch.zeros(images_num, dtype=int)
        if args.gpu is not None:
            gt_image_text = gt_image_text.cuda(args.gpu, non_blocking=True)
        img_text_loss = loss_img(full_mat, gt_image_text)

        # image to text
        txt_logits = i_t_5[:, torch.arange(images_num), pos_idx].t()
        ground_truth_text_image = torch.arange(logits_per_text.shape[1]).long()
        if args.gpu is not None:
            ground_truth_text_image = ground_truth_text_image.cuda(args.gpu, non_blocking=True)
        total_loss = args.soft_max_mil * (img_text_loss +
                                       loss_txt(txt_logits, ground_truth_text_image)) / 2
        return total_loss, tb_loss
    return total_loss, tb_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, loss_func=get_loss):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.train()
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
    if args.distributed and sampler is not None:
        if isinstance(sampler, list):
            [sam_i.set_epoch(epoch) for sam_i in sampler]
        else:
            sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()
        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
        data_time = time.time() - end
        m = model.module if args.distributed or args.dp else model
        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss, tb_loss = loss_func(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss, tb_loss = loss_func(model, images, texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)
        batch_time = time.time() - end
        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.
            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_imgs = tb_loss['images']
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
            for name, val in log_imgs.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_image(name, val.unsqueeze(0), timestep)
        end = time.time()
