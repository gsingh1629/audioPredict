import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader, random_split

# --- 1. CONFIGURATION (Phase 2 Optimization) ---
DATA_DIR = "iot_audio_dataset"
MODEL_SAVE_PATH = "models/rugged_audio_watcher_v4.pth"
# Use the duration from your latest data_gen script
DURATION = 3.2          
SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 100             
LEARNING_RATE = 0.001
N_MELS = 128            # High vertical resolution for speech [cite: 18, 19]
N_FFT = 2048            # High frequency resolution for sharp beeps [cite: 17]
HOP_LENGTH = 512

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 2. FOCAL LOSS (Phase 3: Handling "Hard" Examples) ---
# This puts higher weight on samples the model is struggling with [cite: 28, 29]
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        return loss.mean()

# --- 3. RESIDUAL ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DeepAudioWatcher(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(DeepAudioWatcher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- 4. DATASET ENGINE WITH SPECAUGMENT ---
class RuggedAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate, duration, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        self.num_samples = int(sample_rate * duration)
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        ).to(device)

        # SpecAugment: Randomly masks parts of the spectrogram to prevent 'cheating'
        self.time_mask = T.TimeMasking(time_mask_param=35)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)

        for cls in self.classes:
            cp = os.path.join(root_dir, cls)
            for f in os.listdir(cp):
                if f.endswith('.wav'):
                    self.samples.append((os.path.join(cp, f), self.class_to_idx[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, _ = librosa.load(path, sr=16000)
        waveform = torch.from_numpy(y).float().unsqueeze(0)
        
        # Ensure exact length consistency
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        spec = self.mel_transform(waveform.to(device))
        spec = T.AmplitudeToDB()(spec)

        # Apply SpecAugment only during training to force the model to 'learn' the sounds
        if self.is_training:
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)
            
        return spec, label

# --- 5. EXECUTION ---
def run_rugged_training():
    os.makedirs("models", exist_ok=True)
    full_dataset = RuggedAudioDataset(DATA_DIR, 16000, DURATION)
    
    # Init input shape
    sample_spec, _ = full_dataset[0]
    input_shape = sample_spec.shape
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Mark the validation set so it doesn't use SpecAugment
    val_ds.dataset.is_training = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DeepAudioWatcher(len(full_dataset.classes), input_shape).to(device)
    
    # Phase 3: Using Focal Loss instead of Cross-Entropy [cite: 28, 29]
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print(f"🚀 Starting Rugged Training on {len(full_dataset.classes)} classes...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                _, preds = torch.max(model(specs), 1)
                correct += (preds == labels).sum().item()
        
        val_acc = 100 * correct / val_size
        scheduler.step(val_acc)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

    # --- 6. CONFUSION MATRIX ---
    print("\n📊 Generating Confusion Matrix...")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            _, preds = torch.max(model(specs), 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_dataset.classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.savefig("confusion_matrix_rugged.png")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Success! Saved matrix and model to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_rugged_training()