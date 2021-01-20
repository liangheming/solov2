import math
import torch

import torch.nn.functional as F
from torch import nn
from nets import resnet
from nets.common import FrozenBatchNorm2d
from utils.mask_utils import center_of_mass, matrix_nms
from losses.commons import focal_loss, dice_loss


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(FPN, self).__init__()
        self.latent_layers = list()
        self.out_layers = list()
        for channels in in_channels:
            self.latent_layers.append(nn.Conv2d(channels, out_channels, 1, 1, bias=bias))
            self.out_layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias))
        self.latent_layers = nn.ModuleList(self.latent_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        self.max_pooling = nn.MaxPool2d(1, 2)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs):
        num_layers = len(xs)
        for i in range(num_layers):
            xs[i] = self.latent_layers[i](xs[i])
        for i in range(num_layers):
            layer_idx = num_layers - i - 1
            if i == 0:
                xs[layer_idx] = self.out_layers[layer_idx](xs[layer_idx])
            else:
                d_l = nn.UpsamplingBilinear2d(size=xs[layer_idx].shape[-2:])(xs[layer_idx + 1])
                xs[layer_idx] = self.out_layers[layer_idx](d_l + xs[layer_idx])
        xs.append(self.max_pooling(xs[-1]))
        return xs


default_cfg = {
    "num_cls": 80,
    "backbone": "resnet18",
    "pretrained": True,
    "reduction": False,
    "norm_layer": None,
    "fpn_channels": 256,
    "ins_channels": 512,
    "ins_num_convs": 4,
    "ins_norm": "GN",
    "ins_use_coord": True,
    "kernel_nums": 256,
    "grid_nums": [40, 36, 24, 16, 12],
    "mask_channels": 128,
    "mask_norm": "GN",
    "strides": [8., 8., 16., 32., 32.],
    "scale_range": [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
    "alpha": 0.25,
    "gamma": 2.0,
    "dice_weight": 3.0,
    "focal_weight": 1.0,

    "score_thresh": 0.1,
    "mask_thresh": 0.5,
    "nms_pre": 500,
    "nms_sigma": 2,
    "nms_kernel": "gaussian",
    "update_threshold": 0.05,
    "max_per_img": 100
}


class SOLOv2RepeatConvs(nn.Module):
    def __init__(self, in_channels, inner_channels, num_convs=4, use_coord=True, norm='GN'):
        super(SOLOv2RepeatConvs, self).__init__()
        convs = list()
        for i in range(num_convs):
            if i == 0:
                if use_coord:
                    chnn = in_channels + 2
                else:
                    chnn = in_channels
            else:
                chnn = inner_channels
            convs.append(nn.Conv2d(chnn, inner_channels,
                                   kernel_size=3, stride=1, padding=1, bias=norm is None
                                   ))
            if norm == 'GN':
                convs.append(nn.GroupNorm(32, inner_channels))
            convs.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class SOLOv2InsHead(nn.Module):
    def __init__(self,
                 grid_nums,
                 in_channels=256,
                 inner_channels=512,
                 num_cls=80,
                 kernel_nums=256,
                 num_convs=4,
                 use_coord=True,
                 norm="GN",

                 ):
        super(SOLOv2InsHead, self).__init__()
        self.grid_nums = grid_nums
        self.cate_tower = SOLOv2RepeatConvs(
            in_channels=in_channels,
            inner_channels=inner_channels,
            num_convs=num_convs,
            norm=norm,
            use_coord=False
        )
        self.kernel_tower = SOLOv2RepeatConvs(
            in_channels=in_channels,
            inner_channels=inner_channels,
            num_convs=num_convs,
            norm=norm,
            use_coord=use_coord
        )
        self.cate_pred = nn.Conv2d(inner_channels, num_cls, 3, 1, 1)
        self.kernel_pred = nn.Conv2d(inner_channels, kernel_nums, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        assert len(features) == len(self.grid_nums)
        cate_pred = list()
        kernel_pred = list()

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1],
                                     device=ins_kernel_feat.device,
                                     dtype=ins_kernel_feat.dtype)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2],
                                     device=ins_kernel_feat.device,
                                     dtype=ins_kernel_feat.dtype)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.grid_nums[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=True)
            cate_feat = kernel_feat[:, :-2, :, :]
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))
        return cate_pred, kernel_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self,
                 layer_nums=4,
                 in_channels=256,
                 inner_channels=128,
                 kernel_nums=256,
                 norm="GN"):
        super(SOLOv2MaskHead, self).__init__()
        self.layer_nums = layer_nums
        self.convs_all_levels = nn.ModuleList()
        for i in range(layer_nums):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    in_channels, inner_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, inner_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    chn = in_channels + 2 if i == 3 else in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, inner_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, inner_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    inner_channels, inner_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, inner_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)
            self.convs_all_levels.append(convs_per_level)
        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                inner_channels, kernel_nums,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, kernel_nums),
            nn.ReLU(inplace=True)
        )
        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        assert len(features) == self.layer_nums, \
            print("The number of input features should be equal to the supposed level.")
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.layer_nums):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1,
                                         mask_feat.shape[-1],
                                         device=mask_feat.device,
                                         dtype=mask_feat.dtype)
                y_range = torch.linspace(-1, 1,
                                         mask_feat.shape[-2],
                                         device=mask_feat.device,
                                         dtype=mask_feat.dtype)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
                # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)
        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred


class SOLOv2Assign(object):
    def __init__(self, grid_nums, scale_range, sigma=0.2):
        self.grid_nums = grid_nums
        self.scale_range = scale_range
        self.sigma = sigma

    def __call__(self, mask_feat_size, targets):
        """
        :param mask_feat_size: feature shape of mask_pred (h,w)
        :param targets: dict contain "target","batch_len","mask"
        :return:
        """
        mask_list = targets['mask'].split(targets['batch_len'])
        box_list = targets['target'].split(targets['batch_len'])
        match_idx_list = list()
        for mask, cls_box in zip(mask_list, box_list):
            match_idx = self.assign_single_instance(mask, cls_box, mask_feat_size)
            match_idx_list.append(match_idx)
        return match_idx_list

    def assign_single_instance(self, mask, cls_box, mask_feat_size):
        """
        :param mask:
        :param cls_box:
        :param mask_feat_size:
        :return:
        """
        device = cls_box.device
        gt_box = cls_box[:, 1:]
        gt_areas = ((gt_box[:, 2:] - gt_box[:, :2]) ** 2).sum(-1).sqrt()
        match_idx_for_instance = list()
        for (lower_bound, upper_bound), num_grid in zip(self.scale_range, self.grid_nums):
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()
            match_idx = torch.ones([num_grid, num_grid], device=device) * -1
            if len(hit_indices) == 0:
                match_idx_for_instance.append(match_idx)
                continue
            selected_box = gt_box[hit_indices]
            selected_mask = mask[hit_indices]
            half_ws = 0.5 * (selected_box[:, 2] - selected_box[:, 0]) * self.sigma
            half_hs = 0.5 * (selected_box[:, 3] - selected_box[:, 1]) * self.sigma
            center_ws, center_hs = center_of_mass(selected_mask)
            valid_mask_flags = selected_mask.sum(dim=-1).sum(dim=-1) > 0

            for ind, half_h, half_w, center_h, center_w, valid_mask_flag in zip(hit_indices,
                                                                                half_hs,
                                                                                half_ws,
                                                                                center_hs,
                                                                                center_ws,
                                                                                valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)

                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        match_idx[i, j] = ind
            match_idx_for_instance.append(match_idx)
        if len(match_idx_for_instance) != 0:
            match_idx_for_instance = torch.cat([m.view(-1) for m in match_idx_for_instance])
        # if len(match_idx_for_instance):
        #     print((match_idx_for_instance >= 0).sum(), len(gt_box))
        return match_idx_for_instance


class SOLOv2(nn.Module):
    def __init__(self, **kwargs):
        super(SOLOv2, self).__init__()
        self.cfg = {**default_cfg, **kwargs}
        self.alpha = self.cfg['alpha']
        self.gamma = self.cfg['gamma']
        self.dice_weight = self.cfg['dice_weight']
        self.focal_weight = self.cfg['focal_weight']
        self.backbone = getattr(resnet, self.cfg['backbone'])(
            pretrained=self.cfg['pretrained'],
            reduction=self.cfg['reduction'],
            norm_layer=self.cfg['norm_layer'])
        self.fpn = FPN(in_channels=self.backbone.inner_channels,
                       out_channels=self.cfg['fpn_channels'])

        self.ins_head = SOLOv2InsHead(
            grid_nums=self.cfg['grid_nums'],
            num_cls=self.cfg['num_cls'],
            kernel_nums=self.cfg['kernel_nums'],
            in_channels=self.cfg['fpn_channels'],
            inner_channels=self.cfg['ins_channels'],
            num_convs=self.cfg['ins_num_convs'],
            norm=self.cfg['ins_norm'],
            use_coord=self.cfg['ins_use_coord']
        )

        self.mask_head = SOLOv2MaskHead(
            layer_nums=4,
            in_channels=self.cfg['fpn_channels'],
            inner_channels=self.cfg['mask_channels'],
            kernel_nums=self.cfg['kernel_nums'],
            norm=self.cfg['mask_norm']
        )
        self.assign = SOLOv2Assign(grid_nums=self.cfg['grid_nums'],
                                   scale_range=self.cfg['scale_range'])

    def forward(self, x, valid_size, targets=None):
        x = self.backbone(x)
        x = self.fpn(x)
        ins_feature = self.split_feats(x)
        cate_pred, kernel_pred = self.ins_head(ins_feature)
        ret = dict()
        mask_pred = self.mask_head(x[:-1])
        if self.training:
            match_idx_list = self.assign(mask_feat_size=mask_pred.size()[-2:], targets=targets)
            cls_loss, mask_loss, match_num = self.compute_loss(cate_pred,
                                                               kernel_pred,
                                                               mask_pred,
                                                               match_idx_list,
                                                               targets)
            ret['cls_loss'] = cls_loss
            ret['mask_loss'] = mask_loss
            ret['match_num'] = match_num
        else:
            predicts = self.post_process(cate_pred, kernel_pred, mask_pred, valid_size)
            ret['predicts'] = predicts
        return ret

    @torch.no_grad()
    def post_process(self, cate_pred, kernel_pred, mask_pred, valid_size):
        """
        :param cate_pred:
        :param kernel_pred:
        :param mask_pred:
        :param valid_size:[(w,h),...]
        :return:
        """
        b, c = cate_pred[0].shape[:2]
        k = kernel_pred[0].shape[1]
        cate_pred_flat = torch.cat([cp.view(b, c, -1) for cp in cate_pred], dim=-1).sigmoid()
        kernel_pred_flat = torch.cat([kp.view(b, k, -1) for kp in kernel_pred], dim=-1)
        device = cate_pred_flat.device
        ret = list()
        for single_cat_pred, single_kernel_pred, single_mask_pred, single_size in zip(cate_pred_flat,
                                                                                      kernel_pred_flat,
                                                                                      mask_pred,
                                                                                      valid_size):
            inds = single_cat_pred > self.cfg['score_thresh']
            cate_scores = single_cat_pred[inds]
            if len(cate_scores) == 0:
                ret.append((torch.zeros(size=(0, 6), device=device),
                            torch.zeros(size=(0, single_size[1], single_size[0]), device=device)))
                continue
            inds = inds.nonzero(as_tuple=False)
            cate_labels = inds[:, 0]
            single_kernel_pred = single_kernel_pred[:, inds[:, 1]]
            size_trans = cate_labels.new_tensor(self.cfg['grid_nums']).pow(2).cumsum(0)
            strides = single_kernel_pred.new_ones(size_trans[-1])
            n_stage = len(self.cfg['grid_nums'])
            strides[:size_trans[0]] *= self.cfg['strides'][0]
            for ind_ in range(1, n_stage):
                strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.cfg['strides'][ind_]
            strides = strides[inds[:, 1]]
            seg_preds = F.conv2d(single_mask_pred[None, ...],
                                 single_kernel_pred.T[..., None, None],
                                 stride=1).squeeze(0).sigmoid()
            seg_masks = seg_preds > self.cfg['mask_thresh']
            sum_masks = seg_masks.sum((1, 2)).float()
            keep = sum_masks > strides
            if keep.sum() == 0:
                ret.append((torch.zeros(size=(0, 6), device=device),
                            torch.zeros(size=(0, single_size[1], single_size[0]), device=device)))
                continue
            seg_masks = seg_masks[keep, ...]
            seg_preds = seg_preds[keep, ...]
            sum_masks = sum_masks[keep]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]
            seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
            cate_scores *= seg_scores
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > self.cfg['nms_pre']:
                sort_inds = sort_inds[:self.cfg['nms_pre']]
            seg_masks = seg_masks[sort_inds, :, :]
            seg_preds = seg_preds[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                     sigma=self.cfg['nms_sigma'], kernel=self.cfg['nms_kernel'])
            keep = cate_scores >= self.cfg['update_threshold']
            if keep.sum() == 0:
                ret.append((torch.zeros(size=(0, 6), device=device),
                            torch.zeros(size=(0, single_size[1], single_size[0]), device=device)))
                continue
            seg_preds = seg_preds[keep, :, :]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]

            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > self.cfg['max_per_img']:
                sort_inds = sort_inds[:self.cfg['max_per_img']]
            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]
            seg_masks = F.interpolate(seg_preds.unsqueeze(0),
                                      scale_factor=4,
                                      mode='bilinear',
                                      recompute_scale_factor=True,
                                      align_corners=True).squeeze(0)[:, :single_size[1], :single_size[0]]
            seg_masks = seg_masks > self.cfg['mask_thresh']
            pred_boxes = torch.zeros((seg_masks.size(0), 4), device=device)
            for i in range(seg_masks.size(0)):
                mask = seg_masks[i].squeeze()
                ys, xs = torch.where(mask)
                if len(ys) == 0:
                    pred_boxes[i] = torch.tensor([0, 0, 1, 1], device=device).float()
                else:
                    pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=device).float()
            pred_boxes = torch.cat([pred_boxes, cate_scores[:, None], cate_labels[:, None]], dim=-1)
            ret.append((pred_boxes, seg_masks))
        return ret

    def compute_loss(self, cate_pred, kernel_pred, mask_pred, match_idx_list, targets):
        b, c = cate_pred[0].shape[:2]
        k = kernel_pred[0].shape[1]
        cate_pred_flat = torch.cat([cp.view(b, c, -1) for cp in cate_pred], dim=-1)
        kernel_pred_flat = torch.cat([kp.view(b, k, -1) for kp in kernel_pred], dim=-1)
        mask_list = targets['mask'].split(targets['batch_len'])
        box_list = targets['target'].split(targets['batch_len'])

        cls_pred_list = list()
        cls_target_list = list()

        mask_pred_list = list()
        mask_target_list = list()
        for single_cate_pred, single_kernel_pred, single_mask_pred, match_idx, box_gt, mask_gt in zip(cate_pred_flat,
                                                                                                      kernel_pred_flat,
                                                                                                      mask_pred,
                                                                                                      match_idx_list,
                                                                                                      box_list,
                                                                                                      mask_list):
            if len(match_idx) == 0:
                continue
            match_idx = match_idx.long()
            cls_target = torch.zeros_like(single_cate_pred)
            positive_idx = (match_idx >= 0).nonzero(as_tuple=False).squeeze(-1)
            cls_target[box_gt[match_idx[positive_idx], 0].long(), positive_idx] = 1.
            cls_pred_list.append(single_cate_pred)
            cls_target_list.append(cls_target)
            if len(positive_idx) == 0:
                continue
            kernel_select = single_kernel_pred[:, positive_idx]
            mask_pred_out = F.conv2d(single_mask_pred[None, ...], kernel_select.T[..., None, None], stride=1).squeeze(0)
            mask_pred_list.append(mask_pred_out)

            mask_target = F.interpolate(mask_gt[match_idx[positive_idx]].unsqueeze(1),
                                        scale_factor=0.25,
                                        recompute_scale_factor=True).squeeze(1)
            mask_target_list.append(mask_target)
        cls_pred_list = torch.stack(cls_pred_list)
        cls_target_list = torch.stack(cls_target_list)
        mask_pred_list = torch.cat(mask_pred_list)
        mask_target_list = torch.cat(mask_target_list)
        positive_num = len(mask_pred_list)
        if cls_pred_list.dtype == torch.float16:
            cls_pred_list = cls_pred_list.float()
        if mask_pred_list.dtype == torch.float16:
            mask_pred_list = mask_pred_list.float()
        cls_loss = focal_loss(cls_pred_list.sigmoid(), cls_target_list).sum() / (positive_num + 1)
        mask_loss = dice_loss(mask_pred_list.sigmoid(), mask_target_list).mean()
        return cls_loss * self.focal_weight, mask_loss * self.dice_weight, positive_num

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0],
                              scale_factor=0.5,
                              mode='bilinear',
                              align_corners=True,
                              recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear', align_corners=True))


if __name__ == '__main__':
    inp = torch.rand(size=(4, 3, 640, 640))
    solov2 = SOLOv2()
    out = solov2(inp, None)
    # print(out.shape)
    for o in out:
        print(o.shape)
