import copy
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmcv
from mmcv.runner.fp16_utils import force_fp32
from mmcv.cnn import normal_init
from mmdet.core import (bbox2roi, multi_apply, merge_aug_proposals, anchor_inside_flags, bbox2result,
                        bbox_flip, bbox_overlaps, images_to_levels, levels_to_images, unmap, build_assigner,
                        bbox2distance, distance2bbox, reduce_mean)

from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from ssod.datasets.pipelines.rand_aug import visualize_bboxes

from .multi_stream_detector import MultiSteamDetector
from .utils import (Transform2D, filter_invalid, filter_invalid_classwise, 
    concat_all_gather, filter_invalid_scalewise, get_pseudo_label_quality, resize_image)

from torch.utils.tensorboard import SummaryWriter

import random
import time 
import ipdb

@DETECTORS.register_module()
class PseCo_RetinaNet(MultiSteamDetector):
    """ Quality of pseudo boxes is indicated by prediction variance 
        from multiple anchors.
    """
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(PseCo_RetinaNet, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

            self.register_buffer("precision", torch.zeros(1))
            self.register_buffer("recall", torch.zeros(1))

            self.use_MSL = self.train_cfg.use_MSL
            self.use_PCV = self.train_cfg.use_PCV

    def forward_train(self, imgs, img_metas, **kwargs):
        super().forward_train(imgs, img_metas, **kwargs)
        kwargs.update({"img": imgs})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        loss = {}
    
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            
            sup_loss = self.forward_sup_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        
        if "unsup_student" in data_groups:
            unsup_loss = self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"])
            unsup_loss = weighted_loss(
                unsup_loss,
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        return loss

    def forward_sup_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        forward training process for the labeled data. 
        """
        x = self.extract_feat(img, self.student, start_lvl=1)
        
        losses = self.student.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def foward_unsup_train(self, teacher_data, student_data):
        teacher_img = teacher_data["img"]
        student_img = student_data["img"]

        img_metas_teacher = teacher_data["img_metas"]
        img_metas_student = student_data["img_metas"]

        # for analysis use only
        gt_bboxes, gt_labels = teacher_data["gt_bboxes"], teacher_data["gt_labels"]   
        if len(img_metas_student) > 1:
            tnames = [meta["filename"] for meta in img_metas_teacher]
            snames = [meta["filename"] for meta in img_metas_student]
            tidx = [tnames.index(name) for name in snames]
            teacher_img = teacher_img[torch.Tensor(tidx).to(teacher_img.device).long()]
            img_metas_teacher = [img_metas_teacher[idx] for idx in tidx]
        
        with torch.no_grad():
            det_bboxes, det_labels = self.extract_teacher_info(
                teacher_img, 
                img_metas_teacher)
        gt_bboxes_ori = copy.deepcopy(gt_bboxes)
        
        pseudo_bboxes = self.convert_bbox_space(
                        img_metas_teacher, img_metas_student, det_bboxes)
        gt_bboxes = self.convert_bbox_space(
                        img_metas_teacher, img_metas_student, gt_bboxes)
        pseudo_labels = det_labels

        # student model forward
        feats = self.extract_feat(student_img, self.student, start_lvl=1)   

        if self.use_MSL:
            img_ds = resize_image(student_img)
            feats_V2 = self.extract_feat(img_ds, self.student, start_lvl=0)
        
        losses = self.unsup_bbox_head(
                            student_img,
                            feats, 
                            feats_V2 if self.use_MSL else None,
                            img_metas_student, 
                            pseudo_bboxes, 
                            pseudo_labels,  
                            GT_bboxes=gt_bboxes,
                            GT_labels=gt_labels)
        
        losses["precision"] = self.precision
        losses["recall"] = self.recall

        return losses

    def unsup_bbox_head(self,
                        img,
                        feat,
                        feat_V2,
                        img_metas,
                        pseudo_bboxes,
                        pseudo_labels,
                        GT_bboxes=None,
                        GT_labels=None):  
        
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold)     
        # for analysis only
        precision, recall = get_pseudo_label_quality(
                        gt_bboxes, gt_labels, GT_bboxes, GT_labels)
        self.precision = 0.9 * self.precision + 0.1 * precision
        self.recall = 0.9 * self.recall + 0.1 * recall
        
        outs = self.student.bbox_head(feat) 
        
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.unsup_head_loss(*loss_inputs)
        
        if feat_V2 is not None:
            outs_V2 = self.student.bbox_head(feat_V2)
            loss_inputs_V2 = outs_V2 + (gt_bboxes, gt_labels, img_metas)
            losses_V2 = self.unsup_head_loss(*loss_inputs_V2)
            for key, val in losses_V2.items():
                losses[key + "_V2"] = val
        
        # set cls weight to 0.5
        for key, val in losses.items():
            if "cls" in key:
                if isinstance(val, list):
                    losses[key] = [item * 0.5 for item in val]

        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def unsup_head_loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.student.bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        cls_reg_targets = self.unsup_head_get_targets(
            anchor_list,
            valid_flag_list,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.student.bbox_head.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def unsup_head_get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            flat_bbox_preds,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.student.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = flat_bbox_preds[inside_flags, :]
        
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]

        assign_result = self.student.bbox_head.assigner.assign(
            anchors, num_level_anchors_inside, 
            gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.student.bbox_head.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.student.bbox_head.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        if len(pos_inds) > 0:
            if not self.student.bbox_head.reg_decoded_bbox:
                pos_bbox_targets = self.student.bbox_head.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            
            label_weights[pos_inds] = 1.0
           
            """ compute regression weights according to PCV """
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_decoded_bboxes = self.student.bbox_head.bbox_coder.decode(
                sampling_result.pos_bboxes, pos_bbox_preds
            )
            
            IoUs = bbox_overlaps(pos_decoded_bboxes, 
                          sampling_result.pos_gt_bboxes, 
                          is_aligned=True)
            if self.use_PCV:
                pos_bbox_weights = self.compute_PCV(IoUs, pos_assigned_gt_inds)
                bbox_weights[pos_inds] = pos_bbox_weights

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.student.bbox_head.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def unsup_head_get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    cls_score_list,
                    bbox_pred_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    unmap_outputs=True):
        
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        
        concat_cls_score_list = levels_to_images(cls_score_list, chn=self.student.bbox_head.num_classes)
        concat_bbox_pred_list = levels_to_images(bbox_pred_list, chn=4)

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self.unsup_head_get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            num_level_anchors_list,
            concat_bbox_pred_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
    
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def compute_PCV(self, IoUs, pos_assigned_gt_inds):
        bbox_weights = IoUs.new_zeros(IoUs.shape[0], 4)
        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            idx = (gt_ind == pos_assigned_gt_inds).nonzero().reshape(-1)
            bbox_weights[idx] = IoUs[idx].mean()
        return bbox_weights

    def extract_feat(self, img, model, start_lvl=0):
        if start_lvl == 0:
            feats = model.extract_feat(img)[:-1]
        
        elif start_lvl == 1:
            feats = model.extract_feat(img)[1:]
        
        return feats

    def extract_teacher_info(self, img, img_metas):
        feat = self.extract_feat(img, self.teacher, start_lvl=1)
        results_list = self.teacher.bbox_head.simple_test(feat, img_metas, rescale=False)
        pseudo_bboxes, pseudo_labels = [], []

        for det_bbox, det_label in results_list:
            det_bbox = det_bbox.to(feat[0].device)
            pseudo_bboxes.append(det_bbox if det_bbox.shape[0] > 0 else det_bbox.new_zeros(0, 5))
            pseudo_labels.append(det_label.to(feat[0].device))

        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
       
        pseudo_bboxes, pseudo_labels, _ = list(
            zip(
                *[
                    filter_invalid(
                        bboxes,
                        labels,
                        bboxes[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for bboxes, labels in zip(
                        pseudo_bboxes, pseudo_labels
                    )
                ]
            )
        )

        return pseudo_bboxes, pseudo_labels

    def forward_test(self, imgs, img_metas, **kwargs):

        return super(MultiSteamDetector, self).forward_test(imgs, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        model: TwoStageDetector = getattr(self, 'model')
        return model.aug_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentation."""
        
        model = self.model(**kwargs)
        assert model.with_bbox, 'Bbox head must be implemented.'
        feat = self.extract_feat(img, model, start_lvl=1)
        results_list = model.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, model.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]
    
    def convert_bbox_space(self, img_metas_A, img_metas_B, bboxes_A):
        """ 
            function: convert bboxes_A from space A into space B
            Parameters: 
                img_metas: list(dict); bboxes_A: list(tensors)
        """
        transMat_A = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                                for meta in img_metas_A]
        transMat_B = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                            for meta in img_metas_B]
        M = self._get_trans_mat(transMat_A, transMat_B)
        bboxes_B = self._transform_bbox(
            bboxes_A,
            M,
            [meta["img_shape"] for meta in img_metas_B],
        )
        return bboxes_B

