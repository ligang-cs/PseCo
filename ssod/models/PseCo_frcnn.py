import copy
import os.path as osp
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmcv
from mmcv.runner.fp16_utils import force_fp32
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmdet.core import bbox2roi, multi_apply, merge_aug_proposals, bbox_mapping, bbox_mapping_back, bbox_overlaps, build_assigner
from mmdet.models import BaseDetector, TwoStageDetector, DETECTORS, build_detector
from mmdet.models.builder import build_loss

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.datasets.pipelines.rand_aug import visualize_bboxes

from .multi_stream_detector import MultiSteamDetector
from .utils import (Transform2D, filter_invalid, filter_invalid_classwise, concat_all_gather, 
                   filter_invalid_scalewise, resize_image, get_pseudo_label_quality)
import random
import time 
import os
import ipdb

@DETECTORS.register_module()
class PseCo_FRCNN(MultiSteamDetector):
    """ PseCo on FR-CNN.
    """
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(PseCo_FRCNN, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            
            self.register_buffer("precision", torch.zeros(1))
            self.register_buffer("recall", torch.zeros(1))
            
            # initialize assignment to build condidate bags
            self.PLA_iou_thres = self.train_cfg.get("PLA_iou_thres", 0.4)
            initial_assigner_cfg=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=self.PLA_iou_thres,       
                neg_iou_thr=self.PLA_iou_thres,
                match_low_quality=False,
                ignore_iof_thr=-1)
            self.initial_assigner = build_assigner(initial_assigner_cfg)
            self.PLA_candidate_topk = self.train_cfg.PLA_candidate_topk
            
            self.use_teacher_proposal = self.train_cfg.use_teacher_proposal
            self.use_MSL = self.train_cfg.use_MSL

        if self.student.roi_head.bbox_head.use_sigmoid:
            self.use_sigmoid = True
        else:
            self.use_sigmoid = False
        
        self.num_classes = self.student.roi_head.bbox_head.num_classes

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

    def extract_feat(self, img, model, start_lvl=0):
        """Directly extract features from the backbone+neck."""
        assert start_lvl in [0, 1], \
            f"start level {start_lvl} is not supported."
        x = model.backbone(img)
        # global feature -- [p2, p3, p4, p5, p6, p7]
        if model.with_neck:
            x = model.neck(x)
        if start_lvl == 0:
            return x[:-1]
        elif start_lvl == 1:
            return x[1:]

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
        losses = dict()
        # high resolution
        x = self.extract_feat(img, self.student, start_lvl=1)
        # RPN forward and loss
        if self.student.with_rpn:
            proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                              self.student.test_cfg.rpn)
            rpn_losses, proposal_list = self.student.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RCNN forward and loss 
        roi_losses = self.student.roi_head.forward_train(x, img_metas, proposal_list,
                                                gt_bboxes, gt_labels,
                                                gt_bboxes_ignore, gt_masks,
                                                **kwargs)
        losses.update(roi_losses)
       
        return losses

    def foward_unsup_train(self, teacher_data, student_data):
        teacher_img = teacher_data["img"]
        student_img = student_data["img"]

        img_metas_teacher = teacher_data["img_metas"]
        img_metas_student = student_data["img_metas"]

        gt_bboxes, gt_labels = teacher_data["gt_bboxes"], teacher_data["gt_labels"]   
        if len(img_metas_student) > 1:
            tnames = [meta["filename"] for meta in img_metas_teacher]
            snames = [meta["filename"] for meta in img_metas_student]
            tidx = [tnames.index(name) for name in snames]
            teacher_img = teacher_img[torch.Tensor(tidx).to(teacher_img.device).long()]
            img_metas_teacher = [img_metas_teacher[idx] for idx in tidx]
        
        det_bboxes, det_labels, tea_proposals_tuple = self.extract_teacher_info(
                                teacher_img, img_metas_teacher)
        tea_proposals, tea_feats = tea_proposals_tuple
        tea_proposals_copy = copy.deepcopy(tea_proposals)    # proposals before geometry transform
        
        pseudo_bboxes = self.convert_bbox_space(img_metas_teacher, 
                         img_metas_student, det_bboxes)
        tea_proposals = self.convert_bbox_space(img_metas_teacher, 
                         img_metas_student, tea_proposals) 
        gt_bboxes = self.convert_bbox_space(img_metas_teacher, 
                         img_metas_student, gt_bboxes)  
       
        pseudo_labels = det_labels
        
        loss = {}
        # RPN stage
        feats = self.extract_feat(student_img, self.student, start_lvl=1)
        stu_rpn_outs, rpn_losses = self.unsup_rpn_loss(
                feats, pseudo_bboxes, pseudo_labels, img_metas_student)
        loss.update(rpn_losses)
        
        if self.use_MSL:
            # construct View 2 to learn feature-level scale invariance
            img_ds = resize_image(student_img)   # downsampled images
            feats_ds = self.extract_feat(img_ds, self.student, start_lvl=0)
            _, rpn_losses_ds = self.unsup_rpn_loss(feats_ds, 
                                    pseudo_bboxes, pseudo_labels, 
                                    img_metas_student)
            for key, value in rpn_losses_ds.items():
                loss[key + "_V2"] = value 

        # RCNN stage
        """ obtain proposals """
        if self.use_teacher_proposal:
            proposal_list = tea_proposals

        else :
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *stu_rpn_outs, img_metas_student, cfg=proposal_cfg
            )
        
        """ obtain teacher predictions for all proposals """
        with torch.no_grad():
            rois_ = bbox2roi(tea_proposals_copy)
            tea_bbox_results = self.teacher.roi_head._bbox_forward(
                             tea_feats, rois_)
        
        teacher_infos = {
            "imgs": teacher_img,
            "cls_score": tea_bbox_results["cls_score"].sigmoid() if self.use_sigmoid \
                else tea_bbox_results["cls_score"][:, :self.num_classes].softmax(dim=-1),
            "bbox_pred": tea_bbox_results["bbox_pred"],
            "feats": tea_feats,
            "img_metas": img_metas_teacher,
            "proposal_list": tea_proposals_copy}
       
        rcnn_losses = self.unsup_rcnn_cls_loss(
                            feats,
                            feats_ds if self.use_MSL else None,
                            img_metas_student, 
                            proposal_list, 
                            pseudo_bboxes, 
                            pseudo_labels,  
                            GT_bboxes=gt_bboxes,
                            GT_labels=gt_labels,
                            teacher_infos=teacher_infos)

        loss.update(rcnn_losses)
        loss["precision"] = self.precision
        loss["recall"] = self.recall
        
        return loss

    def unsup_rpn_loss(self, stu_feats, pseudo_bboxes, pseudo_labels, img_metas):
        stu_rpn_outs = self.student.rpn_head(stu_feats)
        # rpn loss 
        gt_bboxes_rpn = []
        for bbox, label in zip(pseudo_bboxes, pseudo_labels):
            bbox, label, _ = filter_invalid(
                bbox[:, :4],
                label=label,
                score=bbox[
                    :, 4
                ],  # TODO: replace with foreground score, here is classification score,
                thr=self.train_cfg.rpn_pseudo_threshold,
                min_size=self.train_cfg.min_pseduo_box_size,
            )
            gt_bboxes_rpn.append(bbox) 

        stu_rpn_loss_inputs = stu_rpn_outs + ([bbox.float() for bbox in gt_bboxes_rpn], img_metas)
        rpn_losses = self.student.rpn_head.loss(*stu_rpn_loss_inputs)
        return stu_rpn_outs, rpn_losses

    def unsup_rcnn_cls_loss(self,
                        feat,
                        feat_V2,
                        img_metas,
                        proposal_list,
                        pseudo_bboxes,
                        pseudo_labels,
                        GT_bboxes=None,
                        GT_labels=None,
                        teacher_infos=None):  
        
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold)

        # quality of pseudo label
        precision, recall = get_pseudo_label_quality(
                        gt_bboxes, gt_labels, GT_bboxes, GT_labels)
        self.precision = 0.9 * self.precision + 0.1 * precision
        self.recall = 0.9 * self.recall + 0.1 * recall
        
        sampling_results = self.prediction_guided_label_assign(
                    img_metas,
                    proposal_list,
                    gt_bboxes,
                    gt_labels,
                    teacher_infos=teacher_infos)
       
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        labels = bbox_targets[0]

        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
       
        bbox_weights = self.compute_PCV(
                bbox_results["bbox_pred"], 
                labels, 
                selected_bboxes,
                pos_gt_bboxes_list, 
                pos_assigned_gt_inds_list)
        bbox_weights_ = bbox_weights.pow(2.0)
        pos_inds = (labels >= 0) & (labels < self.student.roi_head.bbox_head.num_classes)
        if pos_inds.any():
            reg_scale_factor = bbox_weights.sum() / bbox_weights_.sum()
        else:
            reg_scale_factor = 0.0

        # Focal loss
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *(bbox_targets[:3]),
            bbox_weights_,
            reduction_override="none",
        )
        
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = reg_scale_factor * loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0) 
        
        if feat_V2 is not None:
            bbox_results_V2 = self.student.roi_head._bbox_forward(feat_V2, rois)
            loss_V2 = self.student.roi_head.bbox_head.loss(
                bbox_results_V2["cls_score"],
                bbox_results_V2["bbox_pred"],
                rois,
                *(bbox_targets[:3]),
                bbox_weights_,
                reduction_override="none",
            )
            
            loss["loss_cls_V2"] = loss_V2["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
            loss["loss_bbox_V2"] = reg_scale_factor * loss_V2["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0) 
            if "acc" in loss_V2:
                loss["acc_V2"] = loss_V2["acc"]

        # print scores of positive proposals (analysis only)
        tea_cls_score = teacher_infos["cls_score"]
        num_proposal = [proposal.shape[0] for proposal in proposal_list]
        tea_cls_score_list = tea_cls_score.split(num_proposal, dim=0)   # tensor to list
        tea_pos_score = [] 
        for score, pos in zip(tea_cls_score_list, pos_inds_list):
            tea_pos_score.append(score[pos])
        tea_pos_score = torch.cat(tea_pos_score, dim=0)
        
        with torch.no_grad():
            if pos_inds.any():
                max_score = tea_pos_score[torch.arange(tea_pos_score.shape[0]), labels[pos_inds]].float()
                pos_score_mean = max_score.mean()
                pos_score_min = max_score.min()
                
            else:
                max_score = tea_cls_score.sum().float() * 0
                pos_score_mean = tea_cls_score.sum().float() * 0
                pos_score_min = tea_cls_score.sum().float() * 0

        loss["tea_pos_score_mean"] = pos_score_mean
        loss["tea_pos_score_min"] = pos_score_min
        loss['cls_score_thr'] = torch.tensor(self.train_cfg.cls_pseudo_threshold, 
                                             dtype=torch.float, 
                                             device=labels.device)
        loss["pos_number"] = pos_inds.sum().float()

        return loss


    def extract_teacher_info(self, img, img_metas):
        feat = self.extract_feat(img, self.teacher, start_lvl=1)
       
        proposal_cfg = self.teacher.train_cfg.get(
            "rpn_proposal", self.teacher.test_cfg.rpn
        )
        rpn_out = list(self.teacher.rpn_head(feat))
        proposal_list = self.teacher.rpn_head.get_bboxes(
            *rpn_out, img_metas, cfg=proposal_cfg
        )
        
        # teacher proposals
        proposals = copy.deepcopy(proposal_list)

        proposal_list, proposal_label_list = \
            self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, 
            self.teacher.test_cfg.rcnn, 
            rescale=False
        )   # obtain teacher predictions

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list

        return det_bboxes, proposal_label_list, \
            (proposals, feat)

    @torch.no_grad()
    def compute_PCV(self, 
                      bbox_preds, 
                      labels, 
                      proposal_list, 
                      pos_gt_bboxes_list, 
                      pos_assigned_gt_inds_list):
        """ Compute regression weights for each proposal according 
            to Positive-proposal Consistency Voting (PCV). 
        
        Args:
            bbox_pred (Tensors): bbox preds for proposals.
            labels (Tensors): assigned class label for each proposals. 
                0-79 indicate fg, 80 indicates bg.
            propsal_list tuple[Tensor]: proposals for each image.
            pos_gt_bboxes_list, pos_assigned_gt_inds_list tuple[Tensor]: label assignent results 
        
        Returns:
            bbox_weights (Tensors): Regression weights for proposals.
        """

        nums = [_.shape[0] for _ in proposal_list]
        labels = labels.split(nums, dim=0)
        bbox_preds = bbox_preds.split(nums, dim=0)
    
        bbox_weights_list = []

        for bbox_pred, label, proposals, pos_gt_bboxes, pos_assigned_gt_inds in zip(
                    bbox_preds, labels, proposal_list, pos_gt_bboxes_list, pos_assigned_gt_inds_list):

            pos_inds = ((label >= 0) & 
                        (label < self.student.roi_head.bbox_head.num_classes)).nonzero().reshape(-1)
            bbox_weights = proposals.new_zeros(bbox_pred.shape[0], 4)
            pos_proposals = proposals[pos_inds]
            if len(pos_inds):
                pos_bbox_weights = proposals.new_zeros(pos_inds.shape[0], 4)
                pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), -1, 4)[
                                pos_inds, label[pos_inds]
                            ]
                decoded_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                        pos_proposals, pos_bbox_pred)
                
                gt_inds_set = torch.unique(pos_assigned_gt_inds)
                
                IoUs = bbox_overlaps(
                    decoded_bboxes,
                    pos_gt_bboxes,
                    is_aligned=True)
    
                for gt_ind in gt_inds_set:
                    idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
                    if idx_per_gt.shape[0] > 0:
                        pos_bbox_weights[idx_per_gt] = IoUs[idx_per_gt].mean()
                bbox_weights[pos_inds] = pos_bbox_weights
               
            bbox_weights_list.append(bbox_weights)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        
        return bbox_weights

    @torch.no_grad()
    def prediction_guided_label_assign(
                self,
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                teacher_infos,
                gt_bboxes_ignore=None,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        # get teacher predictions (including cls scores and bbox ious)       
        tea_proposal_list = teacher_infos["proposal_list"]
        tea_cls_score_concat = teacher_infos["cls_score"]
        tea_bbox_pred_concat = teacher_infos["bbox_pred"]
        num_per_img = [_.shape[0] for _ in tea_proposal_list]
        tea_cls_scores = tea_cls_score_concat.split(num_per_img, dim=0)
        tea_bbox_preds = tea_bbox_pred_concat.split(num_per_img, dim=0)

        decoded_bboxes_list = []
        for bbox_preds, cls_scores, proposals in zip(tea_bbox_preds, tea_cls_scores, tea_proposal_list):
            pred_labels = cls_scores.max(dim=-1)[1]
            
            bbox_preds_ = bbox_preds.view(
                bbox_preds.size(0), -1, 
            4)[torch.arange(bbox_preds.size(0)), pred_labels]
            
            decode_bboxes = self.student.roi_head.bbox_head.bbox_coder.decode(
                        proposals, bbox_preds_) 
            decoded_bboxes_list.append(decode_bboxes)
        
        decoded_bboxes_list = self.convert_bbox_space(
                                teacher_infos['img_metas'], 
                                img_metas, 
                                decoded_bboxes_list)

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.initial_assigner.assign( 
                decoded_bboxes_list[i], 
                gt_bboxes[i], 
                gt_bboxes_ignore[i], 
                gt_labels[i])
            
            gt_inds = assign_result.gt_inds
            pos_inds = torch.nonzero(gt_inds > 0, as_tuple=False).reshape(-1)

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds]        
            pos_labels = gt_labels[i][pos_assigned_gt_inds]
            
            tea_pos_cls_score = tea_cls_scores[i][pos_inds]
           
            tea_pos_bboxes = decoded_bboxes_list[i][pos_inds]
            ious = bbox_overlaps(tea_pos_bboxes, gt_bboxes[i])
            
            wh = proposal_list[i][:, 2:4] - proposal_list[i][:, :2]
            areas = wh.prod(dim=-1)
            pos_areas = areas[pos_inds]
            
            refined_gt_inds = self.assignment_refinement(gt_inds, 
                                       pos_inds, 
                                       pos_assigned_gt_inds, 
                                       ious, 
                                       tea_pos_cls_score, 
                                       pos_areas, 
                                       pos_labels)
    
            assign_result.gt_inds = refined_gt_inds + 1
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                                assign_result,
                                proposal_list[i],
                                gt_bboxes[i],
                                gt_labels[i])
            sampling_results.append(sampling_result)
        return sampling_results

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds, 
                             ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions 
        # on each image 
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0], ), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)
        
        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]
            target_scores = cls_score[pos_idx_per_gt, target_labels]
            target_areas = areas[pos_idx_per_gt]
            target_IoUs = ious[pos_idx_per_gt, gt_ind]
            
            cost = (target_IoUs * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)
            
            candidate_topk = min(pos_idx_per_gt.shape[0], self.PLA_candidate_topk)   
            topk_ious, _ = torch.topk(target_IoUs, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)      
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[
                target_areas[sort_idx] > 0
            ]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]
            
            refined_pos_gt_inds[pos_idx_per_gt] = pos_assigned_gt_inds[pos_idx_per_gt]
        
        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds

    def forward_test(self, imgs, img_metas, **kwargs):

        return super(MultiSteamDetector, self).forward_test(imgs, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        model: TwoStageDetector = getattr(self, 'model')
        return model.aug_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        
        model = self.model(**kwargs)
        assert model.with_bbox, 'Bbox head must be implemented.'
        
        x = self.extract_feat(img, model, start_lvl=1)

        if proposals is None:
            proposal_list = model.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return model.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

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
        
