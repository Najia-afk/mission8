"""
Data Loader for Flipkart E-commerce Dataset

Provides train/val/test splits with proper augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import json


class FlipkartDataset(Dataset):
    """
    Flipkart e-commerce product image dataset.
    
    Structure:
        dataset/
            category1/
                img1.jpg
                img2.jpg
            category2/
                ...
    """
    
    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.25,
        random_state: int = 42
    ):
        """
        Args:
            root_dir: Path to dataset root
            split: 'train', 'val', or 'test'
            transform: Image transforms
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Collect all images and labels
        self.images = []
        self.labels = []
        self.class_names = []
        
        self._load_dataset()
        
        # Create splits
        self._create_splits(val_ratio, test_ratio, random_state)
        
        # Select appropriate split
        self._select_split(split)
        
        print(f"[FlipkartDataset] {split}: {len(self.images)} samples, "
              f"{len(self.class_names)} classes")
    
    def _load_dataset(self):
        """Load all images and labels from directory structure."""
        categories = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        self.class_names = [c.name for c in categories]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        all_images = []
        all_labels = []
        
        for category in categories:
            label = self.class_to_idx[category.name]
            for img_path in category.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    all_images.append(str(img_path))
                    all_labels.append(label)
        
        self._all_images = all_images
        self._all_labels = all_labels
    
    def _create_splits(
        self, 
        val_ratio: float, 
        test_ratio: float, 
        random_state: int
    ):
        """Create train/val/test splits with stratification."""
        # First split: train+val vs test
        train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
            self._all_images,
            self._all_labels,
            test_size=test_ratio,
            random_state=random_state,
            stratify=self._all_labels
        )
        
        # Second split: train vs val
        val_size = val_ratio / (1 - test_ratio)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_val_imgs,
            train_val_labels,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        self._splits = {
            'train': (train_imgs, train_labels),
            'val': (val_imgs, val_labels),
            'test': (test_imgs, test_labels)
        }
    
    def _select_split(self, split: str):
        """Select the appropriate data split."""
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        
        self.images, self.labels = self._splits[split]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get count of samples per class."""
        counts = {}
        for label in self.labels:
            class_name = self.class_names[label]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


def get_transforms(
    split: str = 'train',
    input_size: Tuple[int, int] = (224, 224),
    augmentation_strength: str = 'medium'
) -> transforms.Compose:
    """
    Get image transforms for a given split.
    
    Args:
        split: 'train', 'val', or 'test'
        input_size: Target image size
        augmentation_strength: 'light', 'medium', or 'strong'
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        if augmentation_strength == 'light':
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
            ])
        elif augmentation_strength == 'medium':
            return transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:  # strong
            return transforms.Compose([
                transforms.Resize((int(input_size[0] * 1.2), int(input_size[1] * 1.2))),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2)
            ])
    else:
        # Validation and test: no augmentation
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])


class FlipkartDataLoader:
    """
    Data loader manager for Flipkart dataset.
    
    Provides train, validation, and test data loaders with proper
    augmentation and batching.
    """
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        input_size: Tuple[int, int] = (224, 224),
        num_workers: int = 4,
        augmentation_strength: str = 'medium',
        val_ratio: float = 0.15,
        test_ratio: float = 0.25,
        random_state: int = 42
    ):
        """
        Args:
            data_dir: Path to dataset
            batch_size: Batch size for all loaders
            input_size: Input image size
            num_workers: Number of data loading workers
            augmentation_strength: Augmentation level for training
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        
        # Create transforms
        self.train_transform = get_transforms('train', input_size, augmentation_strength)
        self.eval_transform = get_transforms('val', input_size)
        
        # Create datasets
        self.train_dataset = FlipkartDataset(
            data_dir, 'train', self.train_transform,
            val_ratio, test_ratio, random_state
        )
        self.val_dataset = FlipkartDataset(
            data_dir, 'val', self.eval_transform,
            val_ratio, test_ratio, random_state
        )
        self.test_dataset = FlipkartDataset(
            data_dir, 'test', self.eval_transform,
            val_ratio, test_ratio, random_state
        )
        
        # Store metadata
        self.class_names = self.train_dataset.class_names
        self.num_classes = len(self.class_names)
        
        print(f"\n[FlipkartDataLoader] Loaded dataset:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {self.class_names}\n")
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader with shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all three data loaders."""
        return (
            self.get_train_loader(),
            self.get_val_loader(),
            self.get_test_loader()
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance.
        
        Returns inverse frequency weights.
        """
        counts = self.train_dataset.get_class_counts()
        total = sum(counts.values())
        
        weights = []
        for class_name in self.class_names:
            count = counts.get(class_name, 1)
            weight = total / (len(self.class_names) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_samples_per_class(self, data_loader: DataLoader, samples_per_class: int = 2):
        """
        Extract sample images, labels, and paths for each class.
        
        Args:
            data_loader: DataLoader to sample from
            samples_per_class: Number of samples per class
        
        Returns:
            images: List of image tensors
            labels: List of labels
            paths: List of image paths
        """
        class_samples = {i: [] for i in range(len(self.class_names))}
        
        # Collect samples
        for images, labels in data_loader:
            for img, label in zip(images, labels):
                label_idx = label.item()
                if len(class_samples[label_idx]) < samples_per_class:
                    class_samples[label_idx].append((img, label_idx))
            
            # Check if we have enough samples
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
        
        # Flatten to lists
        all_images = []
        all_labels = []
        all_paths = []
        
        dataset = data_loader.dataset
        
        for class_idx in sorted(class_samples.keys()):
            for img_tensor, label in class_samples[class_idx]:
                all_images.append(img_tensor)
                all_labels.append(label)
                
                # Find corresponding path from dataset
                matching_indices = [i for i, (path, lbl) in enumerate(dataset.samples) if lbl == class_idx]
                if matching_indices:
                    all_paths.append(str(dataset.samples[matching_indices[len(all_paths) % len(matching_indices)]][0]))
                else:
                    all_paths.append("unknown")
        
        return torch.stack(all_images), all_labels, all_paths
