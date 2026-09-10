import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from training.preprocessing import clean_text

MODEL_NAME = "bert-base-uncased"


def main():
    df = pd.read_excel("data/zoom_classification_dataset_final.xlsx")
    df = df[df["Positive/Negative"] == "Negative"].dropna(subset=["User Review", "Negative Category"])
    df["User Review"] = df["User Review"].apply(clean_text)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Negative Category"])
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        df["User Review"], df["label"], test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    val_dataset = Dataset.from_dict({"text": X_val.tolist(), "label": y_val.tolist()})
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

    training_args = TrainingArguments(
        output_dir="models/bert_issue_model",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_dir="logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train()

    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    print("\n📊 Issue Classification (BERT):")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    model.save_pretrained("models/bert_issue_model")
    tokenizer.save_pretrained("models/bert_issue_model")


if __name__ == "__main__":
    main()
