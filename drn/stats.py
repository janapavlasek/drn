import datetime
import math
import torch
from torch.nn import functional as F

SMALL = 1e-6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.item()


def iou(output, target, eps=1e-7, reduction='mean', smooth=0.0, dims=None):
    """IoU where both are shape (B, H, W) where each pixel is a label."""
    assert reduction in ['mean', 'sum', 'none']

    probs = output.softmax(dim=1)

    bs = target.size(0)
    num_classes = probs.size(1)
    dims = (0, 2)  # TODO don't reduce over batches if reduction is None.

    target = target.view(bs, -1)
    probs = probs.view(bs, num_classes, -1)

    target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
    target = target.permute(0, 2, 1).to(probs.device).to(probs.dtype)  # H, C, H*W

    intersection = torch.sum(probs * target, dim=dims)
    cardinality = torch.sum(probs + target, dim=dims)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth + eps)

    if reduction == 'mean':
        return jaccard_score.mean().item()
    elif reduction == 'sum':
        return jaccard_score.sum().item()
    else:
        return jaccard_score.mean()  # .item()  # .mean(dim=1)


def jaccard_loss(output, target, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        output: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or output of the model.
        target: a tensor of shape [B, H, W].
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.

    Reference: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    probs = output.softmax(dim=1)

    bs = target.size(0)
    num_classes = probs.size(1)
    dims = (0, 2)

    target = target.view(bs, -1)
    probs = probs.view(bs, num_classes, -1)

    target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
    target = target.permute(0, 2, 1)  # H, C, H*W

    intersection = torch.sum(probs * target, dim=dims)
    cardinality = torch.sum(probs + target, dim=dims)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth + eps)
    return jaccard_score


def sec_to_str(delta):
    t = datetime.timedelta(seconds=delta)
    s = str(t)
    return s.split(".")[0] + "s"


if __name__ == '__main__':
    mask = [[0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]]
    target = torch.tensor(mask).unsqueeze(0)
    probs = torch.zeros(1, 3, 5, 5).scatter_(1, target.unsqueeze(1), 100.)

    print(target.shape)
    print(probs)
    # print(probs.sum((2, 3)).shape)
    # print(probs / probs.sum((2, 3)).unsqueeze(-1).unsqueeze(-1))

    # probs = probs / probs.sum((2, 3)).unsqueeze(-1).unsqueeze(-1)
    # jl = jaccard_loss(probs, target)
    iou = iou(probs, target, reduction='none')

    print(iou, iou.shape)
