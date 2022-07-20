import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from ..builder import LOSSES
from .utils import weighted_loss, weight_reduce_loss
import ipdb

def diff_focal_loss(pred, target, weight=None, beta=2.0, hard_filter=False,
                  reduction='mean',
                  avg_factor=None):
    assert len(target) == 3, """target for diff_focal_loss must be a tuple of three elements,
        including category label, student score and teacher score, respectively."""
    label, stu_score, tea_score = target
    # negatives
    if hard_filter:
        scale_factor = torch.clamp(stu_score - tea_score, min=0)
    else:
        scale_factor = stu_score - tea_score
        outlier_scale_factor = torch.min(scale_factor[scale_factor > 0].detach())
        scale_factor[scale_factor < 0] = outlier_scale_factor
        
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    
    # positives
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    if pos.shape[0] > 0:
        pos_label = label[pos].long()

        filter_flags = torch.clamp(tea_score[pos, pos_label] - stu_score[pos, pos_label], min=0) 
        pre_filter_num = torch.tensor(pos.shape[0], device=pred.device, dtype=torch.float)
        post_filter_num = torch.sum(filter_flags > 0).float()
        if hard_filter:
            scale_factor = filter_flags
        else:
            scale_factor = tea_score[pos, pos_label] - stu_score[pos, pos_label]
            if scale_factor[filter_flags > 0].shape[0] > 0:
                outlier_scale_factor = torch.min(scale_factor[filter_flags > 0].detach())
                scale_factor[filter_flags == 0] = outlier_scale_factor
           
        pos_pred = pred[pos, pos_label]
        onelabel = pos_pred.new_ones(pos_pred.shape)
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pos_pred, onelabel, reduction='none') * scale_factor.pow(beta)
    else:
        pre_filter_num, post_filter_num = pred.sum() * 0, pred.sum() * 0
    
    loss = loss.sum(dim=1, keepdim=False)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss, pre_filter_num, post_filter_num 

def robust_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                  reduction='mean',
                  avg_factor=None):
    assert len(target) == 2, """target for tea_guided_focal_loss must be a tuple of two elements,
        including category label and teacher score, respectively."""
    label, tea_score = target
    
    num_classes = pred.size(1)
    target = F.one_hot(label, num_classes=num_classes + 1)
    target = target[:, :num_classes].type_as(pred)

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # focal weight
    pt = tea_score * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + 0.75 *
                    (1 - target)) * pt.pow(gamma)
   
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class DiffFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 hard_filter=True,
                 reduction='mean',
                 loss_weight=1.0):
        super(DiffFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in DFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.hard_filter = hard_filter
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls, pre_filter_number, post_filter_number = diff_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                hard_filter=self.hard_filter,
                reduction=reduction,
                avg_factor=avg_factor)
            loss_cls *= self.loss_weight
        else:
            raise NotImplementedError
        return loss_cls, pre_filter_number, post_filter_number


@LOSSES.register_module()
class RobustFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(RobustFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in DFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = robust_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
            loss_cls *= self.loss_weight
        else:
            raise NotImplementedError
        return loss_cls