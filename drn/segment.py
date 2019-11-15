#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import json
import time
import shutil
import threading
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch.backends import cudnn

from . import drn

from .stats import AverageMeter, sec_to_str, accuracy
from .log_util import DRNLogger
from .io_util import save_output_images, save_colorful_images, save_checkpoint
from . import data_transforms as transforms
from .datasets.seg_list import SegList, SegListMS, CITYSCAPE_PALETTE
from .datasets.progress_tools import ProgressTools, ProgressToolParts

try:
    from modules import batchnormsync
except ImportError:
    pass


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):

    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False, out_dir=".", multi_gpu=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

        self.logger = DRNLogger(__name__, filepath=out_dir)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        # return self.softmax(y), x
        return y, x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'step_t':
        lr = args.lr / (1.0 + epoch // args.step)
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def build_model(arch, num_classes, pretrained=None, out_dir=None, multi_gpu=False):
    model = DRNSeg(arch, num_classes, None, pretrained=True, out_dir=out_dir)

    if pretrained:
        data = torch.load(pretrained)
        if "state_dict" in data:
            state_dict = data["state_dict"]
            data = OrderedDict([(k.replace("module.", ""), v) for k, v in state_dict.items()])

        model_dict = model.state_dict()

        # Remove the last layer, which is called seg.
        skip_last = data["seg.weight"].shape[0] != model_dict["seg.weight"].shape[0]

        # Filter out unnecessary keys.
        filtered = {}
        for k, v in data.items():
            if k in model_dict.keys():
                # If the last layers weren't the same size, don't update those last layer keys.
                if skip_last and ("seg." in k or "up." in k):
                    continue

                filtered[k] = v

        # Overwrite entries in the existing state dict
        model_dict.update(filtered)

        # Load the new state dict
        model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model).cuda()

    return model


def make_data_loader(split, dataset_name, data_dir, list_dir, batch_size=4, workers=1,
                     ms=False, num_transforms=1, random_rotate=None, random_scale=None,
                     crop_sizes=None, scales=None, background=True):
    if split not in ('train', 'test', 'val'):
        raise Exception("Split {} mut be one of ('train', 'test', 'val')".format(split))

    t = []
    t_always = []

    # if dataset_name == "seglist":
    if split == 'train':
        if random_rotate is not None:
            t.append(transforms.RandomRotate(random_rotate))
        if random_scale is not None:
            t.append(transforms.RandomScale(random_scale))

        t.append(transforms.RandomHorizontalFlip())

    if split == 'val' or split == 'train':
        if crop_sizes is not None:
            t.append(transforms.RandomCrop(crop_sizes))

        t.append(transforms.PadToSize())

    t_always.append(transforms.ToTensor())

    if os.path.exists(os.path.join(data_dir, 'info.json')):
        info = json.load(open(os.path.join(data_dir, 'info.json'), 'r'))
        normalize = transforms.Normalize(mean=info['mean'], std=info['std'])

        t_always.append(normalize)

    shuffle = split == 'train'

    # Provide the name of the image if we are testing.
    out_name = split == "test"

    progress_t = [transforms.flip,
                  transforms.rotate,
                  transforms.crop,
                  transforms.resize]
    progress_t = transforms.Compose(progress_t)

    if dataset_name == "seglist":
        if ms and split == 'test' and scales is not None:
            dataset = SegListMS(data_dir, split, transforms.Compose(t + t_always), scales, list_dir=list_dir)
        else:
            dataset = SegList(data_dir, split, transforms.Compose(t + t_always), list_dir=list_dir, out_name=out_name)
    elif dataset_name == "progress_tools":
        if split == "train":
            dataset = ProgressTools(data_dir, split, progress_t, num_transforms=num_transforms,
                                    out_name=out_name, background=background)
        else:
            dataset = ProgressTools(data_dir, split, out_name=out_name, background=background)
    elif dataset_name == "progress_tool_parts":
        if split == "train":
            dataset = ProgressToolParts(data_dir, split, progress_t, num_transforms=num_transforms,
                                        out_name=out_name, background=background)
        else:
            dataset = ProgressToolParts(data_dir, split, out_name=out_name, background=background)
    else:
        raise Exception("Unrecognized dataset {}".format(dataset_name))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=False, drop_last=True
    )

    return loader


def validate(val_loader, model, criterion, logger, eval_score=accuracy, print_freq=50):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    logger.info('Running validation...')

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, target = data[0], data[1]
            if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                   torch.nn.modules.loss.MSELoss]:
                target = target.float()
            img = img.cuda()
            target = target.cuda(async=True)

            # compute output
            output = model(img)[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            if eval_score is not None:
                score.update(eval_score(output, target), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {score.val:.3f} ({score.avg:.3f})'.format(
                                i, len(val_loader), batch_time=batch_time, loss=losses,
                                score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


def train(args, train_loader, model, criterion, optimizer, logger,
          val_loader=None, start_epoch=0, eval_score=accuracy,
          print_freq=10, checkpoint_freq=500, best_val_score=0, vis=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    iteration = 0

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

        for i, (img, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                   torch.nn.modules.loss.MSELoss]:
                target = target.float()

            img = img.cuda()
            target = target.cuda(async=True)

            # compute output
            output = model(img)[0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            if eval_score is not None:
                scores.update(eval_score(output, target), img.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            time_per_epoch = batch_time.avg * len(train_loader)
            epochs_left = args.epochs - epoch - 1
            batches_left = len(train_loader) - i - 1

            time_left = sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
            time_elapsed = sec_to_str(batch_time.sum)
            time_estimate = sec_to_str(args.epochs * time_per_epoch)

            if i % print_freq == 0:
                logger.info('Epoch: [{}/{}] Batch: [{}/{}]  '
                            # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Elapsed: {}  '
                            'ETA: {} / {}  '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                            'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                            'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                                epoch + 1, args.epochs, i, len(train_loader), time_elapsed, time_left, time_estimate,
                                batch_time=batch_time, data_time=data_time, loss=losses, top1=scores))

                if vis is not None:
                    vis.update(iteration, losses.val, scores.val)

            if i % checkpoint_freq == 0 and i > 0:
                checkpoint_path = os.path.join(args.out_dir, 'checkpoint_{}_{}.pth.tar'.format(epoch, i))
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_val_score': best_val_score,
                }, False, filename=checkpoint_path)
                logger.info("Saved checkpoint to {}".format(checkpoint_path))

            iteration += 1

        # evaluate on validation set
        is_best = False
        if val_loader is not None:
            val_score = validate(val_loader, model, criterion, logger)

            is_best = val_score > best_val_score
            best_val_score = max(val_score, best_val_score)

            if vis is not None:
                vis.update(iteration, losses.val, scores.val, val_score)

        checkpoint_path = os.path.join(args.out_dir, 'checkpoint_latest.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_score': best_val_score,
        }, is_best, filename=checkpoint_path)
        logger.info("Saved checkpoint to {}".format(checkpoint_path))

        if (epoch + 1) % 1 == 0:
            history_path = os.path.join(args.out_dir, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
            shutil.copyfile(checkpoint_path, history_path)


def test(eval_data_loader, model, num_classes, logger,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, batch in enumerate(eval_data_loader):
            img, label, name = batch
            data_time.update(time.time() - end)

            img.cuda()
            final = model(img)[0]

            _, pred = torch.max(final, 1)
            pred = pred.cpu().data.numpy()
            batch_time.update(time.time() - end)
            if save_vis:
                save_output_images(pred, name, output_dir)
                save_colorful_images(
                    pred, name, output_dir + '_color',
                    TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
            if has_gt:
                label = label.numpy()
                hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
                logger.info('===> mAP {mAP:.3f}'.format(
                    mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
            end = time.time()
            time_left = sec_to_str(batch_time.avg * (len(eval_data_loader) - i))
            time_elapsed = sec_to_str(batch_time.sum)
            logger.info('Eval: [{0}/{1}]\t'
                        'ETA: {2} / {3}\t'
                        # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(i, len(eval_data_loader), time_elapsed, time_left, batch_time=batch_time,
                                data_time=data_time))

    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_ms(eval_data_loader, model, num_classes, scales, logger,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            # image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)
