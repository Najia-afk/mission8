"""
Multimodal Fusion Model (ViT + BERT)
Based on Willis & Bakos (2025) - Late Fusion Strategy

This script implements multimodal classification combining:
- ViT-B/16: Image encoder (768-dim)
- BERT-base: Text encoder (768-dim)
- Late Fusion: Concatenate + Dense classifier
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, BertModel, ViTModel
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import pickle

# Paths
BASE_DIR = Path('/app') if os.path.exists('/app') else Path('.')
DATA_DIR = BASE_DIR / 'dataset'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

class TextDataProcessor:
    """Process text data from Flipkart CSV"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.image_text_map = self._build_mapping()
    
    def _extract_category(self, tree_str):
        if pd.isna(tree_str):
            return None
        try:
            tree_str = tree_str.strip('[]"')
            parts = tree_str.split('>>')
            if len(parts) >= 2:
                return parts[1].strip().strip('"')
            return parts[0].strip().strip('"')
        except:
            return None
    
    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = ' '.join(text.split())
        return text[:512]
    
    def _build_mapping(self):
        self.df['clean_name'] = self.df['product_name'].apply(self._clean_text)
        self.df['clean_desc'] = self.df['description'].apply(self._clean_text)
        self.df['combined_text'] = self.df['clean_name'] + ' ' + self.df['clean_desc']
        
        mapping = {}
        for _, row in self.df.iterrows():
            img_name = row['image']
            if pd.notna(img_name):
                mapping[img_name] = {
                    'text': row['combined_text'],
                    'name': row['clean_name']
                }
        return mapping


class MultimodalFlipkartDataset(Dataset):
    """Dataset combining images and text"""
    
    def __init__(self, image_dir, image_text_map, tokenizer, transform, class_names):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.class_names = class_names
        self.samples = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    text_data = image_text_map.get(img_name, {})
                    text = text_data.get('text', f"Product from category {class_name}")
                    
                    self.samples.append({
                        'image_path': img_path,
                        'text': text,
                        'label': class_idx
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        encoding = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': sample['label']
        }


class MultimodalFusionModel(nn.Module):
    """
    Late Fusion Model combining ViT (image) and BERT (text) encoders.
    Based on Willis & Bakos (2025) - Late fusion achieves highest accuracy.
    """
    
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        
        # Image encoder (ViT-B/16)
        self.vit_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        
        # Text encoder (BERT)
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_encoder.parameters():
            param.requires_grad = False
        
        # Fusion layers (only trainable part)
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.trainable_params = sum(p.numel() for p in self.fusion.parameters())
    
    def forward(self, images, input_ids, attention_mask):
        with torch.no_grad():
            vit_output = self.vit_encoder(images)
            vit_features = vit_output.last_hidden_state[:, 0]
            
            bert_output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
            bert_features = bert_output.last_hidden_state[:, 0]
        
        fused = torch.cat([vit_features, bert_features], dim=1)
        output = self.fusion(fused)
        
        return output


def train_multimodal(device='cuda'):
    """Train and evaluate multimodal fusion model"""
    
    print("=" * 60)
    print("🔗 MULTIMODAL FUSION (ViT + BERT)")
    print("=" * 60)
    
    # Check for cached model
    cache_path = MODELS_DIR / 'multimodal_fusion.pt'
    
    # Load text data
    csv_path = DATA_DIR / 'flipkart_com-ecommerce_sample_1050.csv'
    if not csv_path.exists():
        print(f"❌ CSV not found at {csv_path}")
        return None
    
    print("📂 Loading text data...")
    processor = TextDataProcessor(csv_path)
    print(f"✅ Loaded {len(processor.image_text_map)} text entries")
    
    # Setup
    print("🤖 Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_names = [
        'Baby_Care', 'Beauty_and_Personal_Care', 'Computers',
        'Home_Decor_and_Festive_Needs', 'Home_Furnishing',
        'Kitchen_and_Dining', 'Watches'
    ]
    
    # Create dataset
    image_dir = str(DATA_DIR / 'flipkart_categories')
    dataset = MultimodalFlipkartDataset(
        image_dir, processor.image_text_map, tokenizer, transform, class_names
    )
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Split
    train_size = int(0.6 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"📊 Split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    
    # Model
    print("🔗 Creating Multimodal Fusion Model...")
    model = MultimodalFusionModel(num_classes=len(class_names)).to(device)
    print(f"   Trainable params: {model.trainable_params:,}")
    
    # Check for cached model
    if cache_path.exists():
        print(f"📂 Loading cached model from {cache_path}")
        checkpoint = torch.load(cache_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        history = checkpoint.get('history', {})
    else:
        print("🚀 Training from scratch...")
        
        optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(15):
            # Train
            model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/15"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = outputs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_acc = accuracy_score(val_labels, val_preds)
            scheduler.step()
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            print(f"  Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2%}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': val_acc,
                    'history': history
                }, cache_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"✅ Training complete. Best val acc: {best_val_acc:.2%}")
    
    # Test evaluation
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            preds = outputs.argmax(dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\n🎯 Multimodal Fusion Results:")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   F1-Score: {f1:.2%}")
    print(f"   Trainable Params: {model.trainable_params:,}")
    
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'trainable_params': model.trainable_params
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    results = train_multimodal(device)
    
    if results:
        print("\n" + "=" * 60)
        print("🏆 MULTIMODAL FUSION COMPLETE")
        print("=" * 60)
        print(f"   Accuracy: {results['accuracy']:.2%}")
        print(f"   F1-Score: {results['f1_score']:.2%}")
