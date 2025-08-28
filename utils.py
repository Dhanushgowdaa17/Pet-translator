# utils.py
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000  # YAMNet requires 16 kHz

def load_audio_for_yamnet(file_obj, sr=TARGET_SR, duration=None):
    """
    Load audio so it's compatible with YAMNet (mono, 16 kHz).
    file_obj can be a file-like object (from Streamlit upload) or a path.
    Returns a 1-D float32 numpy array at sr=16000.
    """
    # soundfile can read file-like objects
    data, file_sr = sf.read(file_obj)
    # If multi-channel, convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Resample if necessary
    if file_sr != sr:
        data = librosa.resample(data.astype('float32'), orig_sr=file_sr, target_sr=sr)
    # Normalize to -1..1 if not already
    if data.dtype != 'float32':
        data = data.astype('float32')
    # If duration specified, trim/pad
    if duration:
        target_len = int(sr * duration)
        if len(data) > target_len:
            data = data[:target_len]
        else:
            pad = target_len - len(data)
            data = np.pad(data, (0, pad))
    return data
