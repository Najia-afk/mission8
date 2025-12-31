"""
Data Loader Module for Mission 8.

This module provides utilities for loading and preprocessing
the Flipkart product dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset


class FlipkartDataLoader:
    """
    Data loader for Flipkart product classification dataset.
    
    Handles loading CSV metadata, image paths, and label encoding.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        random_seed: int = 42
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the Flipkart dataset directory
            random_seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'Images'
        self.random_seed = random_seed
        
        self.df = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.num_classes = 0
        
    def load_data(
        self,
        csv_filename: str = 'flipkart_com-ecommerce_sample_1050.csv'
    ) -> pd.DataFrame:
        """
        Load the dataset CSV.
        
        Args:
            csv_filename: Name of the CSV file
            
        Returns:
            Loaded DataFrame
        """
        csv_path = self.dataset_path / csv_filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} products from {csv_filename}")
        
        return self.df
    
    def extract_categories(
        self,
        category_column: str = 'product_category_tree',
        level: int = 0
    ) -> pd.Series:
        """
        Extract category labels from category tree.
        
        Args:
            category_column: Name of category column
            level: Level of category to extract (0 = main)
            
        Returns:
            Series of extracted categories
        """
        def extract_level(category_tree):
            if pd.isna(category_tree):
                return 'Unknown'
            try:
                categories = category_tree.strip('[]"').split(' >> ')
                if level < len(categories):
                    return categories[level].strip()
                return categories[0].strip()
            except:
                return 'Unknown'
        
        self.df['main_category'] = self.df[category_column].apply(extract_level)
        
        # Encode labels
        self.df['label'] = self.label_encoder.fit_transform(self.df['main_category'])
        self.class_names = self.label_encoder.classes_.tolist()
        self.num_classes = len(self.class_names)
        
        print(f"✓ Extracted {self.num_classes} categories")
        
        return self.df['main_category']
    
    def validate_images(
        self,
        image_column: str = 'image'
    ) -> pd.DataFrame:
        """
        Validate image paths and filter to existing images.
        
        Args:
            image_column: Name of image filename column
            
        Returns:
            Filtered DataFrame
        """
        def get_image_path(row):
            if image_column in self.df.columns:
                img_name = row[image_column]
            else:
                img_name = f"{row['uniq_id']}.jpg"
            return self.images_path / img_name
        
        self.df['image_path'] = self.df.apply(get_image_path, axis=1)
        self.df['image_exists'] = self.df['image_path'].apply(
            lambda x: Path(x).exists()
        )
        
        valid_count = self.df['image_exists'].sum()
        print(f"✓ Valid images: {valid_count} / {len(self.df)}")
        
        # Filter to valid images
        self.df = self.df[self.df['image_exists']].copy()
        
        return self.df
    
    def split_data(
        self,
        test_size: float = 0.25,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df['label'],
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['label'],
            random_state=self.random_seed
        )
        
        print(f"✓ Data split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self) -> pd.Series:
        """
        Get class distribution.
        
        Returns:
            Series with class counts
        """
        return self.df['main_category'].value_counts()
    
    def get_sample_images(
        self,
        n_per_class: int = 3
    ) -> pd.DataFrame:
        """
        Get sample images from each class.
        
        Args:
            n_per_class: Number of samples per class
            
        Returns:
            DataFrame with sample images
        """
        samples = []
        
        for category in self.class_names:
            class_df = self.df[self.df['main_category'] == category]
            sample = class_df.sample(
                min(n_per_class, len(class_df)),
                random_state=self.random_seed
            )
            samples.append(sample)
        
        return pd.concat(samples)


class ImageDataset(Dataset):
    """
    PyTorch Dataset for product images.
    
    Can be used with both ViT and CNN models.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor=None,
        transform=None,
        image_col: str = 'image_path',
        label_col: str = 'label'
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: DataFrame with image paths and labels
            processor: HuggingFace image processor
            transform: torchvision transforms
            image_col: Column name for image paths
            label_col: Column name for labels
        """
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row[self.image_col]
        image = Image.open(image_path).convert('RGB')
        label = row[self.label_col]
        
        # Apply processor or transform
        if self.processor:
            inputs = self.processor(images=image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].squeeze(0)
        elif self.transform:
            pixel_values = self.transform(image)
        else:
            raise ValueError("Either processor or transform must be provided")
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }
