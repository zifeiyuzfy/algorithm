import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

def bbox_iou(box1, box2):
    """
    计算两个边界框之间的 IoU（交并比）
    :param box1: (tensor) [N, 4] 格式为 [x1, y1, x2, y2]
    :param box2: (tensor) [M, 4] 格式为 [x1, y1, x2, y2]
    :return: (tensor) [N, M] 每个 box1 和 box2 之间的 IoU
    """
    N = box1.size(0)
    M = box2.size(0)

    # 计算交集的左上角和右下角坐标
    lt = torch.max(
        box1[:, None, :2],  # [N, 1, 2]
        box2[None, :, :2]   # [1, M, 2]
    )  # [N, M, 2]

    rb = torch.min(
        box1[:, None, 2:],  # [N, 1, 2]
        box2[None, :, 2:]   # [1, M, 2]
    )  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    iou = inter / (area1[:, None] + area2 - inter)  # [N, M]
    return iou

def soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    """
    Soft-NMS实现
    :param boxes:   (tensor) [N,4]
    :param scores:  (tensor) [N]
    :param sigma:   高斯参数
    :param Nt:      IOU阈值
    :param threshold: 分数阈值
    :param method:  0-标准NMS 1-线性加权 2-高斯加权
    :return:        保留的索引
    """
    keep = []
    indexes = torch.arange(0, scores.size(0), dtype=torch.float).cuda() if scores.is_cuda else torch.arange(0,
                                                                                                            scores.size(
                                                                                                                0),
                                                                                                            dtype=torch.float)

    while boxes.numel() > 0:
        # 获取当前最高分框
        max_score_index = torch.argmax(scores)
        cur_box = boxes[max_score_index].unsqueeze(0)
        keep.append(indexes[max_score_index].long())

        # 计算IOU
        ious = bbox_iou(cur_box, boxes)

        # 根据方法调整分数
        if method == 1:  # 线性衰减
            weight = torch.ones_like(ious)
            weight[ious > Nt] = 1 - ious[ious > Nt]
        elif method == 2:  # 高斯衰减
            weight = torch.exp(-(ious * ious) / sigma)
        else:  # 传统NMS
            weight = torch.ones_like(ious)
            weight[ious > Nt] = 0

        scores = scores * weight.squeeze()

        # 过滤低分框
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        indexes = indexes[mask]

    return torch.LongTensor(keep)


def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

class DecodeBox():
    def __init__(self, std, num_classes, use_soft_nms=False, soft_nms_sigma=0.5):
        self.std = std
        self.num_classes = num_classes + 1
        self.use_soft_nms = use_soft_nms  # 新增Soft-NMS开关
        self.soft_nms_sigma = soft_nms_sigma  # 新增高斯参数

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                nms_iou=0.3, confidence=0.5):
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   batch_size, num_rois, 4
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            #   对回归参数进行reshape
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            #----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            #-------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #-------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score   = roi_scores[i]
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = prob[:, c]
                c_confs_m   = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    # keep = nms(
                    #     boxes_to_process,
                    #     confs_to_process,
                    #     nms_iou
                    # )
                    # 修改NMS部分
                    if self.use_soft_nms:
                        keep = soft_nms(boxes_to_process, confs_to_process,
                                        sigma=self.soft_nms_sigma,
                                        Nt=nms_iou,
                                        method=2)
                    else:
                        keep = nms(boxes_to_process, confs_to_process, nms_iou)

                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results