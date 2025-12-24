import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from src.preprocessing import clean_text


from src.utils import LABEL_MAP
import joblib


# ===============================
# LOAD MODEL (CACHE)
# ===============================

def load_lstm():
    model = load_model("models/lstm/lstm_hate_speech_model.h5")
    tokenizer = joblib.load("models/lstm/tokenizer.pkl")
    return model, tokenizer

def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    model.eval()
    return tokenizer, model


def load_bert():
    model_name = "indobenchmark/indobert-base-p1"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    model.eval()
    return tokenizer, model


# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_lstm(text):
    model, tokenizer = load_lstm()

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding="post")

    probs = model.predict(pad)[0]      # shape: (3,)
    pred = np.argmax(probs)

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(p), 3)
            for i, p in enumerate(probs)
        }
    }


def predict_distilbert(text):
    tokenizer, model = load_distilbert()
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(probs[i]), 3)
            for i in range(len(LABEL_MAP))
        }
    }



def predict_bert(text):
    tokenizer, model = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return {
        "label": LABEL_MAP[pred],
        "confidence": {
            LABEL_MAP[i]: round(float(probs[i]), 3)
            for i in range(len(LABEL_MAP))
        }
    }
