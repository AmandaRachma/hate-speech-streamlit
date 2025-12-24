import torch
import numpy as np
import joblib

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from src.utils import LABEL_MAP
from src.preprocessing import clean_text


# ===============================
# LOAD MODEL (CACHE)
# ===============================

def load_lstm():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_model("models/lstm/lstm_hate_speech_model.h5")
    tokenizer = joblib.load("models/lstm/tokenizer.pkl")
    return model, tokenizer, pad_sequences

def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    model.eval()
    return tokenizer, model

def load_bert():
    tokenizer = BertTokenizerFast.from_pretrained("models/bert")
    model = BertForSequenceClassification.from_pretrained("models/bert")
    model.eval()
    return tokenizer, model


# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_lstm(text):
    model, tokenizer, pad_sequences = load_lstm()
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding="post")

    probs = model.predict(pad)[0]
    pred = np.argmax(probs)

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(LABEL_MAP))
        }
    }



def predict_distilbert(text):
    tokenizer, model = load_distilbert()
    text = clean_text(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(LABEL_MAP))
        }
    }


def predict_bert(text):
    tokenizer, model = load_bert()
    text = clean_text(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(LABEL_MAP))
        }
    }
