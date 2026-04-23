import torch
import torchaudio.transforms as T
import numpy as np
import sounddevice as sd
import os
import time
import torch.nn as nn

# --- 1. CONFIGURATION (Must match v4 Training) ---
MODEL_PATH = "models/rugged_audio_watcher_v4.pth"
DATA_DIR = "iot_audio_dataset"
SAMPLE_RATE = 16000
DURATION = 3.2          # Optimized window length [cite: 12, 40]
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Veto System Settings 
BEEP_THRESHOLD = 0.98   # Stricter for short beeps 
VOICE_THRESHOLD = 0.85  # Standard for distinct speech [cite: 38]
MIN_AMPLITUDE = 0.04    # Noise floor gate
OVERLAP_FACTOR = 0.8    # 80% overlap for sliding window [cite: 34]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# --- 2. ARCHITECTURE (Residual Model) ---
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
    def __init__(self, num_classes):
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

# --- 3. INITIALIZATION ---
model = DeepAudioWatcher(len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
).to(device)

# Global variables for the Veto system
last_label = None
audio_buffer = np.zeros(int(SAMPLE_RATE * DURATION))

def predict(audio_data):
    peak = np.max(np.abs(audio_data))
    if peak < MIN_AMPLITUDE:
        return "background", 0

    # Normalization for consistent signal [cite: 12]
    audio_data = audio_data / peak
    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
    
    with torch.no_grad():
        spec = mel_transform(waveform.to(device))
        spec = T.AmplitudeToDB()(spec)
        spec = spec.unsqueeze(0)
        output = model(spec)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)
        return CLASSES[idx.item()], conf.item()

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, last_label
    
    # Update sliding buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.flatten()
    
    label, confidence = predict(audio_buffer)
    
    # Class-Specific Thresholds [cite: 38]
    required_conf = BEEP_THRESHOLD if label == "beep" else VOICE_THRESHOLD
    
    if label != "background" and confidence >= required_conf:
        # Double-Confirmation (The Veto) 
        if label == last_label:
            print(f"✅ EVENT CONFIRMED: {label.upper()} ({confidence*100:.1f}%)")
            last_label = None # Reset to prevent multiple triggers for one sound
        else:
            last_label = label
    else:
        last_label = None

if __name__ == "__main__":
    print(f"--- Audio Watcher Active: Phase 4 ---")
    # Calculate overlap hop in samples [cite: 34]
    step_size = int(SAMPLE_RATE * DURATION * (1 - OVERLAP_FACTOR))
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=step_size, # Moves the sliding window forward by 20%
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")