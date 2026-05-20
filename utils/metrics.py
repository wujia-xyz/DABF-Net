"""
Evaluation Metrics for Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary mask
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary mask
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss"""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better edge localization
    Penalizes prediction errors near boundaries more heavily
    """

    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary mask
        """
        # Get boundary of ground truth
        target_boundary = self._get_boundary(target)

        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred)

        # Calculate distance-weighted loss
        # Points near boundary get higher weight
        boundary_weight = target_boundary * self.theta + (1 - target_boundary) * self.theta0

        # BCE loss weighted by boundary
        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        weighted_bce = (bce * boundary_weight).mean()

        return weighted_bce

    def _get_boundary(self, mask):
        """Extract boundary pixels using morphological operations"""
        # Dilate
        kernel_size = 3
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)

        # Erode
        eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)

        # Boundary = dilated - eroded
        boundary = dilated - eroded

        return boundary


class ComboLoss(nn.Module):
    """Combination of multiple losses for better performance"""

    def __init__(self, use_boundary_loss=False):
        super(ComboLoss, self).__init__()
        # Use pos_weight to handle class imbalance (lesions are small)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.use_boundary_loss = use_boundary_loss

        if use_boundary_loss:
            self.boundary = BoundaryLoss(theta0=1, theta=5)

    def forward(self, pred, target):
        # Move pos_weight to same device as pred
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)

        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)

        if self.use_boundary_loss:
            boundary_loss = self.boundary(pred, target)
            # Emphasize Dice and Boundary for segmentation
            return 0.15 * bce_loss + 0.5 * dice_loss + 0.15 * focal_loss + 0.2 * boundary_loss
        else:
            # Original weights
            return 0.2 * bce_loss + 0.6 * dice_loss + 0.2 * focal_loss


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice Coefficient (F1 Score)

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
        smooth: smoothing factor
    """
    if pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU)

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
        smooth: smoothing factor
    """
    if pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def pixel_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
    """
    if pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()

    correct = (pred == target).float().sum()
    total = torch.numel(target)

    return (correct / total).item()


def precision_recall_f1(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Precision, Recall, and F1 Score

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
        smooth: smoothing factor

    Returns:
        precision, recall, f1
    """
    if pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()

    TP = (pred * target).sum().item()
    FP = ((1 - target) * pred).sum().item()
    FN = (target * (1 - pred)).sum().item()

    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return precision, recall, f1


def sensitivity_specificity(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Sensitivity (Recall/TPR) and Specificity (TNR)

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
        smooth: smoothing factor

    Returns:
        sensitivity, specificity
    """
    if pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()
    target = target.float()

    TP = (pred * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()
    FP = ((1 - target) * pred).sum().item()
    FN = (target * (1 - pred)).sum().item()

    sensitivity = (TP + smooth) / (TP + FN + smooth)  # Same as recall
    specificity = (TN + smooth) / (TN + FP + smooth)

    return sensitivity, specificity


def hausdorff_distance_95(pred, target, threshold=0.5, spacing=(1.0, 1.0)):
    """
    Calculate 95th percentile Hausdorff Distance (HD95)

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary
        spacing: pixel spacing (default 1.0, 1.0)

    Returns:
        hd95: 95th percentile Hausdorff distance
    """
    if isinstance(pred, torch.Tensor):
        if pred.max() > 1:
            pred = torch.sigmoid(pred)
        pred = (pred > threshold).float().cpu().numpy()
        target = target.float().cpu().numpy()

    hd95_list = []

    for b in range(pred.shape[0]):
        pred_mask = pred[b, 0]
        target_mask = target[b, 0]

        # Handle empty masks
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            hd95_list.append(0.0)
            continue
        elif pred_mask.sum() == 0 or target_mask.sum() == 0:
            # Return max possible distance if one is empty
            hd95_list.append(np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2))
            continue

        # Compute distance transforms
        pred_dist = distance_transform_edt(1 - pred_mask, sampling=spacing)
        target_dist = distance_transform_edt(1 - target_mask, sampling=spacing)

        # Get surface distances
        pred_surface = pred_mask.astype(bool)
        target_surface = target_mask.astype(bool)

        # Distance from pred surface to target
        dist_pred_to_target = target_dist[pred_surface]
        # Distance from target surface to pred
        dist_target_to_pred = pred_dist[target_surface]

        # Combine all distances
        all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])

        # Calculate 95th percentile
        hd95 = np.percentile(all_distances, 95)
        hd95_list.append(hd95)

    return np.mean(hd95_list)


def compute_metrics(pred, target, threshold=0.5):
    """
    Compute all metrics at once

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary

    Returns:
        dict of metrics
    """
    dice = dice_coefficient(pred, target, threshold)
    iou = iou_score(pred, target, threshold)
    accuracy = pixel_accuracy(pred, target, threshold)
    precision, recall, f1 = precision_recall_f1(pred, target, threshold)

    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_full_metrics(pred, target, threshold=0.5):
    """
    Compute full metrics including HD95, Sensitivity, Specificity

    Args:
        pred: [B, 1, H, W] prediction logits or probabilities
        target: [B, 1, H, W] ground truth binary mask
        threshold: threshold for converting probabilities to binary

    Returns:
        dict of metrics: DSC, Jac, HD95, Prec, Sen, Spe
    """
    dice = dice_coefficient(pred, target, threshold)
    iou = iou_score(pred, target, threshold)
    precision, recall, f1 = precision_recall_f1(pred, target, threshold)
    sensitivity, specificity = sensitivity_specificity(pred, target, threshold)
    hd95 = hausdorff_distance_95(pred, target, threshold)

    return {
        'dsc': dice,
        'jac': iou,
        'hd95': hd95,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


class MetricTracker:
    """Track metrics over epochs"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'dice': [],
            'iou': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_average(self):
        avg_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
        return avg_metrics

    def get_summary(self):
        summary = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_max'] = np.max(values)
                summary[f'{key}_min'] = np.min(values)
        return summary


if __name__ == '__main__':
    # Test losses and metrics
    B, C, H, W = 4, 1, 256, 256

    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()

    # Test losses
    print("Testing Loss Functions:")
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")

    bce_dice_loss = BCEDiceLoss()
    print(f"BCE+Dice Loss: {bce_dice_loss(pred, target):.4f}")

    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")

    combo_loss = ComboLoss()
    print(f"Combo Loss: {combo_loss(pred, target):.4f}")

    # Test metrics
    print("\nTesting Metrics:")
    metrics = compute_metrics(pred, target)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
