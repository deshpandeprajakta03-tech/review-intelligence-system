import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from app.utils import clean_text

# Load Sentiment Classification Model (RoBERTa)
sentiment_tokenizer = AutoTokenizer.from_pretrained("models/sentiment_model")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment_model")

# Load Issue Classification Model (RoBERTa)
issue_tokenizer = AutoTokenizer.from_pretrained("models/roberta_issue_model")
issue_model = AutoModelForSequenceClassification.from_pretrained("models/roberta_issue_model")

# Fixed issue labels used in training
issue_labels = [
    "App Stability Issue",
    "Audio / Voice Issue",
    "Connection or Access Issue",
    "Performance / Slowness",
    "Recording / Playback Issue",
    "Screen Sharing Issue",
    "User Interface / Usability Issue",
    "Video Quality Issue"
]

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

def predict_issue(text):
    inputs = issue_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = issue_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return issue_labels[pred]

def predict_pipeline(text):
    text = clean_text(text)
    sentiment = predict_sentiment(text)
    if sentiment == "Negative":
        issue = predict_issue(text)
        return {"sentiment": sentiment, "issue": issue}
    else:
        return {"sentiment": sentiment, "issue": None}
