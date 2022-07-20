import torch 

from mmdet.models import HEADS 
from mmdet.models.dense_heads import RetinaHead
from mmdet.core import anchor_inside_flags, unmap, bbox_overlaps
import ipdb

@HEADS.register_module()
class RetinaHead_SSL(RetinaHead):
    
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """
        Performing Positive-proposal Consistency Voting (PCV) and 
        Prediction-guided Label Assignment (PLA) on the unlabeled data. 
        """

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = self.compute_PCV(sampling_result.pos_bboxes, 
                                                         sampling_result.pos_gt_bboxes, 
                                                         sampling_result.pos_assigned_gt_inds)
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
    
    @torch.no_grad()
    def compute_PCV(self, 
                    pos_bboxes, 
                    pos_gt_bboxes, 
                    pos_assigned_gt_inds):
                    
        """ Compute regression weights for each proposal according 
            to Positive-proposal Consistency Voting (PCV). 
        
        Returns:
            bbox_weights (Tensors): Regression weights computed by the PCV.
        """

        pos_bbox_weights = torch.zeros_like(pos_bboxes)
        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        
        IoUs = bbox_overlaps(
            pos_bboxes,
            pos_gt_bboxes,
            is_aligned=True)

        for gt_ind in gt_inds_set:
            idx_per_gt = (pos_assigned_gt_inds == gt_ind).nonzero().reshape(-1)
            if idx_per_gt.shape[0] > 0:
                pos_bbox_weights[idx_per_gt] = IoUs[idx_per_gt].mean()
        ipdb.set_trace()
        return pos_bbox_weights 
