import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from training.preprocessing import clean_text
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

MAX_LEN = 50
BATCH_SIZE = 16
EPOCHS = 6
EMBED_DIM = 128
HIDDEN_DIM = 128

class IssueDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data = [torch.tensor([vocab.get(tok, 1) for tok in text.split()[:MAX_LEN]]) for text in texts]
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True)
    return padded, torch.tensor(labels)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def main():
    df = pd.read_excel("data/zoom_classification_dataset_final.xlsx")
    df = df[df["Positive/Negative"] == "Negative"].dropna(subset=["User Review", "Negative Category"])
    df["User Review"] = df["User Review"].apply(clean_text)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Negative Category"])
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(df["User Review"], df["label"], test_size=0.2, random_state=42)

    all_tokens = [tok for text in X_train for tok in text.split()]
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(Counter(all_tokens).most_common())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    train_ds = IssueDataset(X_train.tolist(), y_train.tolist(), vocab)
    val_ds = IssueDataset(X_val.tolist(), y_val.tolist(), vocab)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y_batch.numpy())

    print("\n📊 Issue Classification (LSTM) Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    torch.save(model.state_dict(), "models/legacy_models/lstm_issue.pth")

if __name__ == "__main__":
    main()
