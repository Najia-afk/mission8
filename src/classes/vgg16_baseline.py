"""
VGG16 Baseline Classifier for Product Classification.

This module provides the baseline CNN model (VGG16 transfer learning)
from Mission 6 for comparison with PanCAN (Panoptic Context Aggregation Networks).

References:
    - Simonyan & Zisserman (2015). "Very Deep Convolutional Networks"
      https://arxiv.org/abs/1409.1556
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class VGG16Classifier:
    """
    VGG16 Transfer Learning classifier for product images.
    
    This class implements the baseline approach from Mission 6
    using VGG16 pretrained on ImageNet.
    
    Attributes:
        num_classes (int): Number of output classes
        image_size (int): Input image size
        model: The Keras model
        history: Training history
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        image_size: int = 224,
        trainable_layers: int = 0
    ):
        """
        Initialize the VGG16 classifier.
        
        Args:
            num_classes: Number of output classes
            class_names: List of class names
            image_size: Input image size (default 224)
            trainable_layers: Number of VGG16 layers to unfreeze
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.image_size = image_size
        self.trainable_layers = trainable_layers
        
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> Model:
        """
        Build the VGG16 transfer learning model.
        
        Returns:
            Keras Model
        """
        # Load pretrained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Optionally unfreeze some layers for fine-tuning
        if self.trainable_layers > 0:
            for layer in base_model.layers[-self.trainable_layers:]:
                layer.trainable = True
        
        # Build classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        
        return model
    
    def compile(
        self,
        learning_rate: float = 1e-4,
        optimizer: str = 'adam'
    ):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate
            optimizer: Optimizer name ('adam' or 'sgd')
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def get_data_generators(
        self,
        train_df,
        val_df,
        batch_size: int = 32,
        augment: bool = True
    ) -> Tuple:
        """
        Create data generators for training and validation.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            batch_size: Batch size
            augment: Whether to use data augmentation
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='main_category',
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='main_category',
            target_size=(self.image_size, self.image_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(
        self,
        train_generator,
        val_generator,
        epochs: int = 20,
        checkpoint_path: Optional[Path] = None
    ):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            checkpoint_path: Path to save best model
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
        
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_generator) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Tuple of (loss, accuracy)
        """
        return self.model.evaluate(test_generator, verbose=1)
    
    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        """
        Predict the class of a single image.
        
        Args:
            image: Image array (224, 224, 3), normalized to [0, 1]
            
        Returns:
            Tuple of (class_id, class_name, confidence)
        """
        # Ensure correct shape
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        probs = self.model.predict(image, verbose=0)
        class_id = np.argmax(probs[0])
        confidence = probs[0][class_id]
        class_name = self.class_names[class_id]
        
        return class_id, class_name, confidence
    
    def predict_batch(
        self,
        images: np.ndarray
    ) -> List[Tuple[int, str, float]]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: Batch of images (N, 224, 224, 3)
            
        Returns:
            List of (class_id, class_name, confidence) tuples
        """
        probs = self.model.predict(images, verbose=0)
        results = []
        
        for prob in probs:
            class_id = np.argmax(prob)
            confidence = prob[class_id]
            class_name = self.class_names[class_id]
            results.append((class_id, class_name, confidence))
        
        return results
    
    def save(self, path: Path):
        """Save the model."""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: Path):
        """Load a saved model."""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from {path}")
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum([
            np.prod(w.shape) 
            for w in self.model.trainable_weights
        ])
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return self.model.count_params()
