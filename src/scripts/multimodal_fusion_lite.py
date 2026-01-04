"""
MULTIMODAL FUSION - LITE VERSION
ViT-B/16 + DistilBERT Late Fusion
Memory-optimized for limited GPU memory
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Clear GPU cache
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Check memory
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"💾 Free GPU memory: {free_mem / 1024**3:.2f} GB")

print("=" * 60)
print("🔗 MULTIMODAL FUSION LITE (ViT + DistilBERT)")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n📂 Loading text data...")
csv_path = "/app/dataset/flipkart_com-ecommerce_sample_1050.csv"
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} text entries")

# Combine text fields
df['combined_text'] = df['product_name'].fillna('') + ' ' + df['description'].fillna('')

# Extract category (first level)
df['category'] = df['product_category_tree'].apply(
    lambda x: x.split('>>')[0].strip(' "[]') if pd.notna(x) else 'Unknown'
)

# Label encode
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])
num_classes = len(label_encoder.classes_)
print(f"📊 Classes: {num_classes}")
for i, cls in enumerate(label_encoder.classes_):
    print(f"   {i}: {cls}")

# ============================================================
# 2. LIGHTWEIGHT TEXT ENCODER (TF-IDF + MLP)
# ============================================================
print("\n🤖 Creating text encoder (TF-IDF based - lightweight)...")
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit TF-IDF on all text
tfidf = TfidfVectorizer(max_features=768, ngram_range=(1, 2))
tfidf.fit(df['combined_text'].values)
print(f"✅ TF-IDF fitted with {len(tfidf.vocabulary_)} features")

# ============================================================
# 3. IMAGE ENCODER (Pretrained CNN - lighter than ViT)
# ============================================================
print("\n🖼️ Loading image encoder (EfficientNet-B0)...")
from torchvision import transforms, models

# Use EfficientNet-B0 (lighter than ViT)
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
efficientnet.classifier = nn.Identity()  # Remove classifier, get 1280-dim features
efficientnet = efficientnet.to(device)
efficientnet.eval()
for param in efficientnet.parameters():
    param.requires_grad = False
print("✅ EfficientNet-B0 loaded (frozen)")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 4. MULTIMODAL DATASET
# ============================================================
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tfidf_vectorizer, transform):
        self.df = dataframe.reset_index(drop=True)
        self.tfidf = tfidf_vectorizer
        self.transform = transform
        self.img_dir = "/app/dataset/flipkart_images_1050"
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text features (TF-IDF)
        text = row['combined_text']
        text_features = torch.tensor(
            self.tfidf.transform([text]).toarray()[0], 
            dtype=torch.float32
        )
        
        # Image
        img_filename = str(row['image']).split('/')[-1] if pd.notna(row['image']) else None
        img_path = os.path.join(self.img_dir, img_filename) if img_filename else None
        
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)
        
        label = row['label']
        return text_features, image, torch.tensor(label, dtype=torch.long)

# ============================================================
# 5. FUSION MODEL
# ============================================================
class MultimodalFusionLite(nn.Module):
    """Late fusion: EfficientNet (1280) + TF-IDF (768) -> classifier"""
    def __init__(self, text_dim=768, image_dim=1280, num_classes=7):
        super().__init__()
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Image projection  
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # Late fusion (concatenation)
        fused = torch.cat([text_proj, image_proj], dim=1)
        return self.classifier(fused)

# Alias for backward compatibility
MultimodalClassifierLite = MultimodalFusionLite

# ============================================================
# 6. PREPARE DATA
# ============================================================
print("\n📊 Preparing datasets...")
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.625, stratify=temp_df['label'], random_state=42)

print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_dataset = MultimodalDataset(train_df, tfidf, img_transform)
val_dataset = MultimodalDataset(val_df, tfidf, img_transform)
test_dataset = MultimodalDataset(test_df, tfidf, img_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ============================================================
# 7. EXTRACT IMAGE FEATURES (batch process to save memory)
# ============================================================
print("\n🖼️ Extracting image features...")

def extract_image_features(loader, model):
    """Extract image features using frozen backbone"""
    all_text_features = []
    all_image_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for text_feat, images, labels in loader:
            images = images.to(device)
            img_feat = model(images).cpu()
            
            all_text_features.append(text_feat)
            all_image_features.append(img_feat)
            all_labels.append(labels)
    
    return (
        torch.cat(all_text_features),
        torch.cat(all_image_features),
        torch.cat(all_labels)
    )

train_text, train_img, train_labels = extract_image_features(train_loader, efficientnet)
val_text, val_img, val_labels = extract_image_features(val_loader, efficientnet)
test_text, test_img, test_labels = extract_image_features(test_loader, efficientnet)

print(f"✅ Features extracted")
print(f"   Train: text {train_text.shape}, image {train_img.shape}")

# Free up memory
del efficientnet
torch.cuda.empty_cache()

# ============================================================
# 8. TRAIN FUSION MODEL
# ============================================================
print("\n🔗 Creating Fusion Model...")
fusion_model = MultimodalFusionLite(
    text_dim=768,
    image_dim=1280,  # EfficientNet-B0 output
    num_classes=num_classes
).to(device)

trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
print(f"📐 Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print("\n🚀 Training Multimodal Fusion...")
num_epochs = 50
best_val_acc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    fusion_model.train()
    train_loss = 0
    train_correct = 0
    
    # Mini-batch training on extracted features
    indices = torch.randperm(len(train_text))
    for i in range(0, len(train_text), batch_size):
        batch_idx = indices[i:i+batch_size]
        text_feat = train_text[batch_idx].to(device)
        img_feat = train_img[batch_idx].to(device)
        labels = train_labels[batch_idx].to(device)
        
        optimizer.zero_grad()
        outputs = fusion_model(text_feat, img_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
    
    train_acc = train_correct / len(train_text)
    
    # Validation
    fusion_model.eval()
    val_correct = 0
    val_loss = 0
    
    with torch.no_grad():
        for i in range(0, len(val_text), batch_size):
            text_feat = val_text[i:i+batch_size].to(device)
            img_feat = val_img[i:i+batch_size].to(device)
            labels = val_labels[i:i+batch_size].to(device)
            
            outputs = fusion_model(text_feat, img_feat)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
    
    val_acc = val_correct / len(val_text)
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(fusion_model.state_dict(), '/app/models/multimodal_fusion_lite.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ Best validation accuracy: {best_val_acc:.4f}")

# ============================================================
# 9. EVALUATION
# ============================================================
print("\n📊 Final Evaluation on Test Set...")
fusion_model.load_state_dict(torch.load('/app/models/multimodal_fusion_lite.pt'))
fusion_model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for i in range(0, len(test_text), batch_size):
        text_feat = test_text[i:i+batch_size].to(device)
        img_feat = test_img[i:i+batch_size].to(device)
        labels = test_labels[i:i+batch_size]
        
        outputs = fusion_model(text_feat, img_feat)
        preds = outputs.argmax(1).cpu()
        
        all_preds.extend(preds.numpy())
        all_true.extend(labels.numpy())

test_acc = accuracy_score(all_true, all_preds)
test_f1 = f1_score(all_true, all_preds, average='weighted')

print(f"\n{'='*60}")
print(f"🎯 MULTIMODAL FUSION LITE RESULTS")
print(f"{'='*60}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test F1 Score: {test_f1:.4f}")

print("\n📋 Classification Report:")
print(classification_report(all_true, all_preds, target_names=label_encoder.classes_))

# ============================================================
# 10. COMPARE WITH BASELINES
# ============================================================
print("\n📊 COMPARISON WITH PREVIOUS MODELS:")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12}")
print("-" * 60)
print(f"{'PanCANLite (CNN)':<25} {'84.79%':<12} {'84.79%':<12}")
print(f"{'VGG16 (Transfer)':<25} {'84.79%':<12} {'84.57%':<12}")
print(f"{'ViT-B/16 (Image only)':<25} {'86.69%':<12} {'86.53%':<12}")
print(f"{'Ensemble (ViT+CNN)':<25} {'88.21%':<12} {'88.04%':<12}")
print("-" * 60)
print(f"{'Multimodal Fusion Lite':<25} {f'{test_acc*100:.2f}%':<12} {f'{test_f1:.4f}':<12}")
print("=" * 60)

improvement = (test_acc - 0.8669) * 100
print(f"\n📈 Improvement over ViT-only: {improvement:+.2f}%")
print(f"📈 Improvement over Ensemble: {(test_acc - 0.8821) * 100:+.2f}%")

# Save results
import json
results = {
    "model": "Multimodal Fusion Lite",
    "architecture": "EfficientNet-B0 (1280-dim) + TF-IDF (768-dim) -> Late Fusion",
    "test_accuracy": float(test_acc),
    "test_f1": float(test_f1),
    "trainable_params": trainable_params,
    "num_classes": num_classes,
    "fusion_type": "late_fusion",
    "text_encoder": "TF-IDF",
    "image_encoder": "EfficientNet-B0"
}

with open('/app/models/multimodal_fusion_lite_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to /app/models/multimodal_fusion_lite_results.json")
