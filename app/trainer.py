import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from datasets import Dataset

def train_sentiment_model(excel_path, output_dir):
    df = pd.read_excel(excel_path)
    df = df[["User Review", "Positive/Negative"]].dropna()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Positive/Negative"])  # Positive=1, Negative=0

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["User Review"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_fn, batched=True)
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    preds = trainer.predict(val_dataset)
    print("📊 Sentiment Model Performance:")
    print(classification_report(val_labels, np.argmax(preds.predictions, axis=1), target_names=label_encoder.classes_))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_issue_model(excel_path, output_dir):
    df = pd.read_excel(excel_path)
    df = df[df["Positive/Negative"] == "Negative"].copy()
    df = df[["User Review", "Negative Category"]].dropna()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Negative Category"])
    num_labels = len(label_encoder.classes_)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["User Review"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_fn, batched=True)
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    preds = trainer.predict(val_dataset)
    print("📊 Issue Classifier Performance:")
    print(classification_report(val_labels, np.argmax(preds.predictions, axis=1), target_names=label_encoder.classes_))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
