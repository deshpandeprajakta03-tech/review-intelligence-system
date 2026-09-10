import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_sentiment_data(df):
    """
    Preprocess data for sentiment classification (Positive/Negative).
    Returns: train_texts, val_texts, train_labels, val_labels
    """
    df = df[["User Review", "Positive/Negative"]].dropna()
    df["User Review"] = df["User Review"].apply(clean_text)
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Positive/Negative"])  # Positive=1, Negative=0
    return train_test_split(df["User Review"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)

def preprocess_issue_data(df):
    """
    Preprocess data for issue classification (only Negative reviews).
    Returns: train_texts, val_texts, train_labels, val_labels, label_encoder
    """
    df = df[df["Positive/Negative"] == "Negative"].copy()
    df = df[["User Review", "Negative Category"]].dropna()
    df["User Review"] = df["User Review"].apply(clean_text)
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Negative Category"])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["User Review"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels, label_encoder
