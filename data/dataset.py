"""
BUSI Ultrasound Segmentation Dataset Loader
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class BUSIDataset(Dataset):
    """BUSI Breast Ultrasound Dataset for Segmentation"""

    def __init__(self, image_paths, mask_paths, transform=None, mode='train'):
        """
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            transform: Albumentations transform
            mode: 'train' or 'val' or 'test'
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Normalize mask to binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask has correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # Convert mask to float
        mask = mask.float()

        return {
            'image': image,
            'mask': mask,
            'image_path': self.image_paths[idx]
        }


def get_train_transforms(image_size=256):
    """Training data augmentation pipeline"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    """Validation data transform pipeline"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def prepare_dataloaders(data_root, batch_size=8, image_size=256, num_workers=4, val_split=0.2, random_seed=42):
    """
    Prepare train and validation dataloaders

    Args:
        data_root: Root directory of BUSI dataset
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of workers for data loading
        val_split: Validation split ratio
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, dataset_info
    """
    image_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'seg')

    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    # Create paired paths
    image_paths = []
    mask_paths = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        # Corresponding mask file
        mask_file = img_file.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)

    print(f"Total samples found: {len(image_paths)}")

    # Split into train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths,
        test_size=val_split,
        random_state=random_seed,
        shuffle=True
    )

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Create datasets
    train_dataset = BUSIDataset(
        train_images, train_masks,
        transform=get_train_transforms(image_size),
        mode='train'
    )

    val_dataset = BUSIDataset(
        val_images, val_masks,
        transform=get_val_transforms(image_size),
        mode='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataset_info = {
        'train_size': len(train_images),
        'val_size': len(val_images),
        'total_size': len(image_paths),
        'image_size': image_size,
    }

    return train_loader, val_loader, dataset_info


if __name__ == '__main__':
    # Test dataset loading
    data_root = '/home/zqq/wujia/ultrasound/datasets/BUSI_lesion_dataset'
    train_loader, val_loader, info = prepare_dataloaders(data_root, batch_size=4)

    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Image: {batch['image'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    print(f"  Mask value range: [{batch['mask'].min():.2f}, {batch['mask'].max():.2f}]")
