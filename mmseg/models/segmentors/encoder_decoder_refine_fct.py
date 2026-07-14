"""EncoderDecoderRefine with Flip-Consistency Training (FCT).

Extends BPR's EncoderDecoderRefine by tripling the training batch with
H-flipped and V-flipped copies and adding a consistency loss that forces
the model's predictions to match across flipped views (after un-flipping).
"""
import torch
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS
from .encoder_decoder_refine import EncoderDecoderRefine


def _hflip(t):
    """Horizontal flip on the last dimension. Works for 3D and 4D tensors."""
    return torch.flip(t, dims=[-1])


def _vflip(t):
    """Vertical flip on the second-to-last dimension. Works for 3D and 4D tensors."""
    return torch.flip(t, dims=[-2])


@SEGMENTORS.register_module()
class EncoderDecoderRefineFCT(EncoderDecoderRefine):
    """BPR EncoderDecoderRefine + Flip-Consistency Training."""

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_coarse_mask=True,
                 output_float=False,
                 fct_weight=0.3,
                 fct_warmup_iters=300,
                 fct_use_vflip=True):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            use_coarse_mask=use_coarse_mask,
            output_float=output_float,
        )
        self.fct_weight = float(fct_weight)
        self.fct_warmup_iters = int(fct_warmup_iters)
        self.fct_use_vflip = bool(fct_use_vflip)
        # Registered as a buffer so the warmup counter is saved in the model
        # state_dict and restored on resume-from. A plain int would reset to 0
        # and replay the (short) warmup ramp after every resume, briefly
        # under-weighting the consistency loss.
        self.register_buffer('_fct_iter', torch.zeros((), dtype=torch.long))

    def _current_fct_weight(self):
        it = int(self._fct_iter)
        if self.fct_warmup_iters <= 0 or it >= self.fct_warmup_iters:
            return self.fct_weight
        return self.fct_weight * (it / float(self.fct_warmup_iters))

    def forward_train(self, img, img_metas, gt_semantic_seg, coarse_mask):
        B = img.shape[0]

        # ----- Build augmented batch -----
        img_h = _hflip(img)
        cm_h = _hflip(coarse_mask)
        gt_h = _hflip(gt_semantic_seg)

        if self.fct_use_vflip:
            img_v = _vflip(img)
            cm_v = _vflip(coarse_mask)
            gt_v = _vflip(gt_semantic_seg)
            img_all = torch.cat([img, img_h, img_v], dim=0)
            cm_all = torch.cat([coarse_mask, cm_h, cm_v], dim=0)
            gt_all = torch.cat([gt_semantic_seg, gt_h, gt_v], dim=0)
            n_views = 3
        else:
            img_all = torch.cat([img, img_h], dim=0)
            cm_all = torch.cat([coarse_mask, cm_h], dim=0)
            gt_all = torch.cat([gt_semantic_seg, gt_h], dim=0)
            n_views = 2

        metas_all = list(img_metas) * n_views

        # ----- Standard BPR forward path -----
        if self.use_coarse_mask:
            cm_input = (cm_all - 0.5) / 0.5
            x_in = torch.cat([img_all, cm_input[:, None, ...]], dim=1)
        else:
            x_in = img_all

        features = self.extract_feat(x_in)

        # Single decode-head forward, reused for BOTH the supervised loss and
        # the consistency term. Calling _decode_head_forward_train and then
        # _decode_head_forward_test would run the head twice on identical
        # features, doubling the head's BatchNorm running-stat updates per
        # iteration (and wasting compute/activation memory).
        seg_logits_raw = self.decode_head(features)

        losses = dict()
        losses.update(add_prefix(
            self.decode_head.losses(seg_logits_raw, gt_all), 'decode'))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                features, metas_all, gt_all)
            losses.update(loss_aux)

        # ----- Flip-consistency loss (from the same logits) -----
        seg_logits = F.interpolate(
            seg_logits_raw, size=img.shape[-2:],
            mode='bilinear', align_corners=self.align_corners)

        if seg_logits.shape[1] >= 2:
            probs = F.softmax(seg_logits, dim=1)[:, 1:2]
        else:
            probs = torch.sigmoid(seg_logits)

        prob_orig = probs[:B]
        prob_h = probs[B:2 * B]
        prob_h_unflipped = _hflip(prob_h)
        cons_h = F.mse_loss(prob_h_unflipped, prob_orig)

        if self.fct_use_vflip:
            prob_v = probs[2 * B:3 * B]
            prob_v_unflipped = _vflip(prob_v)
            cons_v = F.mse_loss(prob_v_unflipped, prob_orig)
            consistency = 0.5 * (cons_h + cons_v)
        else:
            consistency = cons_h

        lam = self._current_fct_weight()
        losses['loss_fct'] = lam * consistency

        self._fct_iter += 1

        return losses
        #python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 tools/train.py configs/bpr/hrnet48_256_fct.py --launcher pytorch --work-dir work_dirs/hrnet48_256_fct
