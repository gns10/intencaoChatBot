from functools import lru_cache
from pathlib import Path
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from src.utils import tokenizeFunc

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "feelings.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# O treinamento atual usa sequências com tamanho 20.
MAX_LEN = 20


@lru_cache(maxsize=1)
def load_artifacts():
    """Carrega modelo e artefatos uma única vez por processo."""
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


def predict_intent(text: str, threshold: float = 0.50):
    """Classifica uma mensagem e retorna intenção, confiança e probabilidades."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("O texto não pode estar vazio.")

    model, tokenizer, label_encoder = load_artifacts()

    treated_text = tokenizeFunc(text)
    sequence = tokenizer.texts_to_sequences([treated_text])
    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probabilities = model.predict(sequence, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    intent = str(label_encoder.inverse_transform([class_index])[0])

    return {
        "intent": intent,
        "confidence": confidence,
        "accepted": confidence >= threshold,
        "probabilities": probabilities,
    }
