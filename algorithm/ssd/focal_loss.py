import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, pos_threshold=0.5):
        """
        Args:
            num_classes (int): 类别数量（不包括背景）
            alpha (float): 平衡正负样本的权重（默认 0.25）
            gamma (float): 调节难易样本的权重（默认 2）
            pos_threshold (float): 正样本的置信度阈值（默认 0.5）
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.pos_threshold = pos_threshold

    def forward(self, y_true, y_pred):
        """
        Args:
            y_true (Tensor): [batch, num_anchors, 4 + num_classes + 1]
            y_pred (tuple): (loc_pred, conf_pred)
                - loc_pred: [batch, num_anchors, 4]
                - conf_pred: [batch, num_anchors, num_classes]
        Returns:
            total_loss (Tensor): 总损失
        """
        # 解包预测值（定位 + 分类）
        loc_pred, conf_pred = y_pred  # y_pred 是元组 (loc_pred, conf_pred)

        # 分割真实值（定位 + 分类 + 掩码）
        loc_truth = y_true[:, :, :4]  # [batch, num_anchors, 4]
        conf_truth = y_true[:, :, 4:-1]  # [batch, num_anchors, num_classes]
        pos_mask = y_true[:, :, -1]  # [batch, num_anchors]（1=正样本，0=负样本）

        # 计算定位损失（仅正样本）
        loc_loss = F.smooth_l1_loss(
            loc_pred * pos_mask.unsqueeze(-1),
            loc_truth * pos_mask.unsqueeze(-1),
            reduction="sum"
        )

        # 计算分类的Focal Loss
        conf_truth_flat = torch.argmax(conf_truth, dim=-1).view(-1)  # [batch * num_anchors]
        conf_pred_flat = conf_pred.view(-1, self.num_classes)  # [batch * num_anchors, num_classes]

        # 计算交叉熵损失（Focal Loss 的基础）
        ce_loss = F.cross_entropy(conf_pred_flat, conf_truth_flat, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = softmax(预测概率)

        # 设置 alpha_t（正样本用 alpha，负样本用 1-alpha）
        alpha_t = torch.where(conf_truth_flat > 0, self.alpha, 1 - self.alpha)
        focal_loss = (alpha_t * (1 - pt) ** self.gamma * ce_loss).sum()

        # 归一化（除以正样本数量）
        num_pos = pos_mask.sum().clamp(min=1.0)
        total_loss = (loc_loss + focal_loss) / num_pos

        return total_loss