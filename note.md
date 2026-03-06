# PhysioNet Heart Sound Model — Colab Training Guide

> Step-by-step setup for training a PhysioNet/CinC 2016 heart sound classifier (normal vs abnormal)

---

## 0) Overview

The challenge dataset contains thousands of phonocardiogram recordings collected from varied environments. The goal is to decide whether a recording should be referred as abnormal or not.

**Pipeline:**

```
PhysioNet wav files → resample/trim/pad → mel spectrogram → CNN → normal vs abnormal
```

This is the best beginner path on free Colab because it is much simpler than doing full S1/S2 segmentation first.

---

## 1) Create a New Colab Notebook

1. Open Colab and create a new notebook
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. If no GPU is available, use CPU temporarily and retry later

---

## 2) Get the Dataset

Use the official PhysioNet/CinC 2016 source. Two options:

### Option A: Upload zip to Google Drive (simplest)

1. Download the dataset zip from PhysioNet to your PC
2. Upload it to Google Drive
3. Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then unzip:

```bash
!unzip -q "/content/drive/MyDrive/physionet2016.zip" -d /content/data
```

### Option B: Download directly in Colab

```bash
!mkdir -p /content/data
%cd /content/data
!wget -O challenge2016.zip "PUT_THE_DIRECT_ZIP_LINK_HERE"
!unzip -q challenge2016.zip
```

If direct download is blocked, use Option A.

---

## 3) Install Python Packages

```bash
!pip -q install librosa soundfile scipy scikit-learn pandas tqdm
```

For PyTorch:

```bash
!pip -q install torch torchvision torchaudio
```

---

## 4) Inspect the Dataset Structure

The training set is organized into databases A–E, with ~3,100 recordings total.

```python
import os
from glob import glob

wav_files = glob('/content/data/**/*.wav', recursive=True)
txt_files = glob('/content/data/**/*.txt', recursive=True)

print("wav:", len(wav_files))
print("txt:", len(txt_files))
print(wav_files[:5])
```

Look for:
- `.wav` audio files
- label/reference text files

---

## 5) Build a Label Table

Target format:

| file_path | label |
|-----------|-------|
| .../a0001.wav | 0 (normal) |
| .../a0002.wav | 1 (abnormal) |

```python
import pandas as pd
import os
from glob import glob

rows = []
ref_files = glob('/content/data/**/*.txt', recursive=True)

for ref in ref_files:
    fname = os.path.basename(ref).lower()
    if 'reference' not in fname and 'annotations' not in fname:
        continue

    with open(ref, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rec_id, label_str = parts[0], parts[1].lower()

            if label_str in ['-1', 'abnormal']:
                label = 1
            elif label_str in ['1', 'normal']:
                label = 0
            else:
                continue

            candidates = glob(f'/content/data/**/{rec_id}.wav', recursive=True)
            if candidates:
                rows.append({'file_path': candidates[0], 'label': label})

df = pd.DataFrame(rows).drop_duplicates()
print(df.head())
print(df['label'].value_counts())
```

> **Note:** Inspect the actual reference files first — naming may differ slightly across folders.

---

## 6) Train / Validation Split

Stratified split to keep normal/abnormal proportions balanced:

```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

print(len(train_df), len(val_df))
```

---

## 7) Convert Audio to Mel Spectrograms

**Settings:**
- Resample: 2000 Hz
- Fixed length: 5 seconds
- Mel bins: 64
- Log-mel scale

```python
import numpy as np
import librosa

SR = 2000
DURATION = 5
N_SAMPLES = SR * DURATION
N_MELS = 64
FMIN = 25
FMAX = 500

def load_audio(file_path, sr=SR, n_samples=N_SAMPLES):
    y, _ = librosa.load(file_path, sr=sr)
    y = y / (np.max(np.abs(y)) + 1e-8)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]
    return y

def audio_to_mel(file_path):
    y = load_audio(file_path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
```

Test one sample:

```python
import matplotlib.pyplot as plt

sample = audio_to_mel(train_df.iloc[0]['file_path'])
print(sample.shape)

plt.figure(figsize=(10, 4))
plt.imshow(sample, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Mel spectrogram")
plt.show()
```

---

## 8) PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class HeartSoundDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = audio_to_mel(row['file_path'])
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = torch.tensor(mel).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(row['label']).long()
        return mel, label

train_ds = HeartSoundDataset(train_df)
val_ds = HeartSoundDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
```

---

## 9) CNN Model

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## 10) Training Loop

```python
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits, y).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return val_loss / len(loader), acc, f1

best_f1 = 0.0

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss, val_acc, val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "/content/best_heartsound_cnn.pth")
```

---

## 11) Save Model to Drive

```bash
!cp /content/best_heartsound_cnn.pth /content/drive/MyDrive/
```

---

## 12) Inference on a Single File

```python
def predict_file(model, file_path):
    model.eval()
    mel = audio_to_mel(file_path)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs.argmax(), probs

label_map = {0: "normal", 1: "abnormal"}
file_path = val_df.iloc[0]['file_path']
pred, probs = predict_file(model, file_path)
print(file_path)
print("Prediction:", label_map[pred], probs)
```

---

## 13) Improvements Roadmap

### A. Better Evaluation
- Confusion matrix, sensitivity, specificity, ROC-AUC

### B. Handle Class Imbalance
The dataset has many more normal than abnormal recordings. Use weighted loss or oversampling:

```python
class_counts = train_df['label'].value_counts().sort_index().values
weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

### C. Data Augmentation
- Small noise, time shift, gain change

### D. Multiple Windows per Recording
- Split into several 5-second windows, classify each, average probabilities

### E. Transfer Learning
- Convert mel to 3 channels → use EfficientNet or ResNet18

---

## 14) Common Colab Problems

| Problem | Solution |
|---------|----------|
| Runtime disconnects | Save checkpoints every epoch; keep outputs in Drive |
| GPU not available | Use CPU for preprocessing, retry later |
| RAM / slow mel generation | Preprocess once and save `.npy` files |

**Preprocessing to `.npy`:**

```python
import os, numpy as np
from tqdm import tqdm

os.makedirs('/content/mels', exist_ok=True)
for i, row in tqdm(df.iterrows(), total=len(df)):
    mel = audio_to_mel(row['file_path'])
    np.save(f"/content/mels/{i}.npy", mel)
```

---

## 15) Recommended Stages

| Stage | Focus |
|-------|-------|
| 1 | Normal vs abnormal — 5s mel spectrogram CNN |
| 2 | Multi-window voting per recording |
| 3 | Add segmentation or cycle-aware features |
| 4 | Combine PCG with ECG/SCG for multimodal work |

---

## 16) Minimum Notebook Sections

1. Mount Drive
2. Install packages
3. Unzip dataset
4. Inspect files
5. Build label dataframe
6. Train/val split
7. Audio preprocessing
8. Dataset / DataLoader
9. Model
10. Train loop
11. Evaluation
12. Save model

---

## 17) Realistic Expectations

A compact CNN on mel spectrograms is feasible on free Colab. The best 2016 challenge methods combined deep learning with handcrafted features or ensemble methods — a tiny CNN alone won't match those, but it's a solid starting point.