import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from training.preprocessing import preprocess_issue_data

# Load dataset
df = pd.read_excel("data/zoom_classification_dataset_final.xlsx")

# Preprocess for issue classification (only Negative reviews)
train_texts, val_texts, train_labels, val_labels, label_encoder = preprocess_issue_data(df)
num_labels = len(label_encoder.classes_)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_fn, batched=True)
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_fn, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

# Training arguments
args = TrainingArguments(
    output_dir="models/roberta_issue_model",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train
trainer.train()

# Evaluate
preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print("📊 Issue Classification Performance:")
print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))

# Save model
model.save_pretrained("models/roberta_issue_model")
tokenizer.save_pretrained("models/roberta_issue_model")
