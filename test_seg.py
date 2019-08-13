
import os
import sys
import argparse

import torch
from torch.backends import cudnn
from torch import nn

from drn import get_logger
from drn.segment import build_model, test, test_ms, train, validate, make_data_loader

logger = None


def train_seg(args):
    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = build_model(args.arch, args.classes, args.pretrained)
    criterion = nn.NLLLoss(ignore_index=255)

    criterion.cuda()

    train_loader = make_data_loader('train', args.dataset, args.data_dir, args.list_dir,
                                    batch_size=args.batch_size, workers=args.workers,
                                    random_rotate=args.random_rotate, random_scale=args.random_scale,
                                    crop_size=args.crop_size)
    val_loader = make_data_loader('val', args.dataset, args.data_dir, args.list_dir,
                                  batch_size=args.batch_size, workers=args.workers,
                                  crop_size=args.crop_size)

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.module.optim_parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    start_epoch = 0
    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    train(args, train_loader, model, criterion, optimizer, val_loader=val_loader,
          start_epoch=start_epoch, best_prec1=best_prec1)


def test_seg(args):
    # batch_size = args.batch_size
    # num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = build_model(args.arch, args.classes, args.pretrained)

    scales = [0.5, 0.75, 1.25, 1.5, 1.75]

    test_loader = make_data_loader(phase, args.dataset, args.data_dir, args.list_dir,
                                   batch_size=args.batch_size, workers=args.workers,
                                   ms=args.ms, scales=scales)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    out_dir = os.path.join(args.out_dir, out_dir)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    logger.info('mAP: {}'.format(mAP))


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default="seglist")
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('--output-dir', default=".", dest="out_dir",
                        help='Directory to store output files.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    # if args.bn_sync:
    #     drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()

    global logger
    logger = get_logger()

    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
