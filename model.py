# model.py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import io

YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
# CSV with class map for YAMNet (AudioSet)
YAMNET_CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

# Load model once
_yamnet_model = None
_class_names = None

def load_yamnet():
    global _yamnet_model, _class_names
    if _yamnet_model is None:
        _yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    if _class_names is None:
        # Fetch class map CSV into memory (tf.keras.utils.get_file will download at runtime)
        path = tf.keras.utils.get_file(
            "yamnet_class_map.csv",
            YAMNET_CLASS_MAP_URL
        )
        df = pd.read_csv(path)
        # 'display_name' column is common
        _class_names = df['display_name'].tolist()
    return _yamnet_model, _class_names

def predict_yamnet(waveform, sr=16000):
    """
    waveform: 1-D numpy float32 at sr=16000
    returns: dict with top classes/scores and averaged score vector
    """
    model, class_names = load_yamnet()
    # YAMNet expects a float32 Tensor of shape [n_samples]
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # Run the model — outputs scores (frame-wise), embeddings, spectrogram
    scores, embeddings, spectrogram = model(waveform_tf)
    # scores: (num_frames, num_classes)
    scores_np = scores.numpy()
    # Average across frames
    mean_scores = np.mean(scores_np, axis=0)
    # top-5 indices
    top_n = 6
    top_idx = mean_scores.argsort()[-top_n:][::-1]
    top = [(class_names[i], float(mean_scores[i])) for i in top_idx]
    return {
        "mean_scores": mean_scores,
        "top": top
    }

# Heuristic mapping from YAMNet classes to our pet-intent categories.
# We'll check if certain keywords appear in predicted class names.
INTENT_CATEGORIES = ["Hungry 🍖", "Wants to Play 🎾", "In Pain 🚑", "Alert ⚡", "Happy ❤️", "Unknown 🤷"]

# Keywords mapping (lowercase) -> intents
KEYWORD_INTENT_MAP = {
    # Pain/distress keywords
    "whine": "In Pain 🚑",
    "whimper": "In Pain 🚑",
    "yelp": "In Pain 🚑",
    "scream": "In Pain 🚑",
    "growl": "In Pain 🚑",
    "howl": "In Pain 🚑",

    # Play / Attention / Excited
    "bark": "Wants to Play 🎾",
    "dog": "Wants to Play 🎾",  # dog sounds often bark/playful
    "pant": "Wants to Play 🎾",
    "squeak": "Wants to Play 🎾",

    # Alert / aggressive / threat
    "alarm": "Alert ⚡",
    "sirens": "Alert ⚡",
    "siren": "Alert ⚡",
    "snarl": "Alert ⚡",
    "growl": "Alert ⚡",

    # Cat-specific
    "meow": "Hungry 🍖",   # heuristic: meows often = seeking food/attention
    "purr": "Happy ❤️",
    "hiss": "Alert ⚡",

    # Bird/chirp
    "chirp": "Happy ❤️",
}

def map_to_intent(top_classes):
    """
    top_classes: list of (class_name, score) sorted desc
    Returns single intent string and short rationale.
    """
    # examine top few classes and pick first matching keyword
    examined = top_classes[:6]
    for name, score in examined:
        low = name.lower()
        for kw, intent in KEYWORD_INTENT_MAP.items():
            if kw in low:
                return intent, f"Detected sound '{name}' (score {score:.2f})."
    # fallback: if any 'dog' / 'cat' present with decent confidence, guess play/happy
    for name, score in examined:
        low = name.lower()
        if "dog" in low and score > 0.1:
            return "Wants to Play 🎾", f"Detected dog-related sound '{name}' (score {score:.2f})."
        if "cat" in low and score > 0.1:
            return "Hungry 🍖", f"Detected cat-related sound '{name}' (score {score:.2f})."
    return "Unknown 🤷", "Could not confidently map detected sound to a pet intent."

def analyze(waveform):
    """
    Full pipeline: run YAMNet, map to intent.
    Returns dict with top classes, mapped_intent, rationale.
    """
    out = predict_yamnet(waveform)
    intent, rationale = map_to_intent(out['top'])
    result = {
        "intent": intent,
        "rationale": rationale,
        "top": out['top']
    }
    return result
