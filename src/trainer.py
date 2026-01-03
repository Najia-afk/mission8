"""
PanCAN Trainer

Training loop with proper validation, early stopping, and checkpointing.
Designed for correct transfer learning: frozen backbone, trainable context modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime
import json
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


class PanCANTrainer:
    """
    Trainer for PanCAN models.
    
    Features:
    - Mixed precision training (with stability safeguards)
    - Gradient clipping
    - Early stopping
    - Model checkpointing
    - Detailed metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        save_dir: Path,
        class_names: List[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        patience: int = 10,
        use_amp: bool = False,  # Disabled by default for numerical stability
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: PanCAN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            save_dir: Directory for checkpoints
            class_names: List of class names
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            num_epochs: Maximum epochs
            patience: Early stopping patience
            use_amp: Use automatic mixed precision (disabled for stability)
            gradient_clip: Gradient clipping value
            label_smoothing: Label smoothing factor
            class_weights: Optional class weights for imbalanced data
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.num_epochs = num_epochs
        self.patience = patience
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        
        # Loss function with label smoothing
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # Optimizer - only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
        
        # AMP scaler (only if enabled)
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Training state
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Print training configuration
        self._print_config(learning_rate, weight_decay)
    
    def _print_config(self, lr: float, wd: float):
        """Print training configuration."""
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {wd}")
        print(f"Max epochs: {self.num_epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient clipping: {self.gradient_clip}")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*60 + "\n")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n[Warning] NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n[Warning] NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*accuracy_score(all_labels, all_preds):.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Evaluating") -> Dict[str, Any]:
        """
        Evaluate model on a data loader.
        
        Args:
            loader: Data loader to evaluate on
            desc: Description for progress bar
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(loader, desc=desc, leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Training history
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate(self.val_loader, "Validation")
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved! (Val Acc: {100*val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation accuracy: {100*self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def test(self, load_best: bool = True) -> Dict[str, Any]:
        """
        Evaluate on test set.
        
        Args:
            load_best: Whether to load best checkpoint first
            
        Returns:
            Test metrics
        """
        if load_best:
            self._load_best_checkpoint()
        
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, "Testing")
        
        # Print results
        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"Accuracy: {100*test_metrics['accuracy']:.2f}%")
        print(f"F1 (macro): {100*test_metrics['f1_macro']:.2f}%")
        print(f"F1 (weighted): {100*test_metrics['f1_weighted']:.2f}%")
        print(f"Precision: {100*test_metrics['precision']:.2f}%")
        print(f"Recall: {100*test_metrics['recall']:.2f}%")
        
        # Classification report
        print(f"\n{'='*60}")
        print("Classification Report")
        print(f"{'='*60}")
        report = classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=self.class_names
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions']
        )
        test_metrics['confusion_matrix'] = cm
        
        return test_metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'class_names': self.class_names
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
    
    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        checkpoint_path = self.save_dir / 'best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("No checkpoint found, using current model")
    
    def save_results(self, test_metrics: Dict[str, Any], filename: str = 'results.json'):
        """Save training results to JSON."""
        results = {
            'training': {
                'best_epoch': self.best_epoch,
                'best_val_acc': float(self.best_val_acc),
                'history': {k: [float(v) for v in vals] for k, vals in self.history.items()}
            },
            'test': {
                'accuracy': float(test_metrics['accuracy']),
                'f1_macro': float(test_metrics['f1_macro']),
                'f1_weighted': float(test_metrics['f1_weighted']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall'])
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_classes': len(self.class_names),
                'class_names': self.class_names
            }
        }
        
        with open(self.save_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.save_dir / filename}")
