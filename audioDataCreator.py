import os
import librosa
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "inputSoundData"
BEEP_FILE = "beep.amr"
# TARGET IoT States only
OTHER_FILES = ["power_off_en.amr", "power_off_hi.amr", "otp_en.amr", "otp_hi.amr", "power_on.amr"]
# Samples used ONLY for the 'background' class
ENV_SAMPLES = ["env1.amr", "env2.amr"] 
OUTPUT_DIR = "iot_audio_dataset"
SAMPLE_RATE = 16000

def get_system_duration(files):
    durations = []
    for f in files + [BEEP_FILE]:
        path = os.path.join(INPUT_DIR, f)
        if os.path.exists(path):
            durations.append(librosa.get_duration(path=path))
    # Standardize to longest file (usually ~3s) 
    return max(3.0, max(durations) if durations else 3.0)

SYSTEM_DURATION = get_system_duration(OTHER_FILES)
print(f"📏 System Window Standardized to: {SYSTEM_DURATION}s")

def save_with_jitter(y_fragment, sr, class_name, filename):
    """Pads to system duration with Random Jitter """
    target_samples = int(sr * SYSTEM_DURATION)
    y_final = np.zeros(target_samples)
    
    # RANDOM JITTER: Place the sound at a random timestamp 
    if len(y_fragment) < target_samples:
        max_start = target_samples - len(y_fragment)
        start_idx = np.random.randint(0, max_start)
        y_final[start_idx : start_idx + len(y_fragment)] = y_fragment
    else:
        y_final = y_fragment[:target_samples]
        
    # Standardize Volume/Normalization [cite: 10]
    if np.max(np.abs(y_final)) > 0:
        y_final = y_final / np.max(np.abs(y_final))

    path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(path, exist_ok=True)
    sf.write(os.path.join(path, filename), y_final, sr)

def generate_rugged_background(count=500):
    """Combines real env samples and Gaussian noise into one 'background' class """
    print(f"🏢 Generating {count} Rugged Background samples...")
    path = os.path.join(OUTPUT_DIR, "background")
    os.makedirs(path, exist_ok=True)
    
    # Load environment sounds
    env_data = []
    for f in ENV_SAMPLES:
        p = os.path.join(INPUT_DIR, f)
        if os.path.exists(p):
            y, _ = librosa.load(p, sr=SAMPLE_RATE)
            env_data.append(y)

    for i in range(count):
        duration_samples = int(SAMPLE_RATE * SYSTEM_DURATION)
        
        # Start with base Gaussian noise [cite: 13]
        noise = 0.01 * np.random.normal(size=duration_samples)
        
        # Mix in a random slice of real environment noise if available [cite: 13, 41]
        if env_data:
            source = env_data[np.random.randint(0, len(env_data))]
            if len(source) > duration_samples:
                start = np.random.randint(0, len(source) - duration_samples)
                noise += source[start : start + duration_samples]
            else:
                noise[:len(source)] += source

        sf.write(os.path.join(path, f"bg_{i:03d}.wav"), noise, SAMPLE_RATE)

def generate_dataset():
    # 1. BEEP: 200 total (fragments scaled to SYSTEM_DURATION) 
    beep_path = os.path.join(INPUT_DIR, BEEP_FILE)
    if os.path.exists(beep_path):
        print("🔔 Processing Beeps with Jitter...")
        y, sr = librosa.load(beep_path, sr=SAMPLE_RATE)
        idx = 0
        # Use fractions of the system duration for fragments [cite: 7, 12]
        for frac in [0.25, 0.5, 0.75, 1.0]:
            for _ in range(50):
                slice_len = int(sr * (SYSTEM_DURATION * frac))
                y_slice = y[:slice_len]
                save_with_jitter(y_slice, sr, "beep", f"beep_f{frac}_{idx:03d}.wav")
                idx += 1

    # 2. OTHERS: 500 total each [cite: 11]
    for fname in OTHER_FILES:
        path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(path): continue
        
        c_name = os.path.splitext(fname)[0]
        print(f"🎙️ Processing {c_name} with fragments...")
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        idx = 0
        
        # Generate varied fragment lengths 
        for frac in [0.4, 0.6, 0.8, 1.0]:
            for _ in range(100):
                slice_len = int(sr * (SYSTEM_DURATION * frac))
                y_slice = y[:slice_len]
                save_with_jitter(y_slice, sr, c_name, f"{c_name}_f{frac}_{idx:03d}.wav")
                idx += 1
        
        # 100 fragments with offset 
        for _ in range(100):
            start_sample = int(sr * 0.5) # 0.5s offset
            y_offset = y[start_sample:]
            save_with_jitter(y_offset, sr, c_name, f"{c_name}_offset_{idx:03d}.wav")
            idx += 1

    # 3. BACKGROUND: 500 samples [cite: 13, 41]
    generate_rugged_background(500)

if __name__ == "__main__":
    generate_dataset()
    print(f"\n✅ Rugged Dataset Complete. Standardized to {SYSTEM_DURATION}s.")