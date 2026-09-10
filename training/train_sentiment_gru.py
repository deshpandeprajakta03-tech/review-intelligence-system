import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from training.preprocessing import clean_text
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

MAX_LEN = 50
BATCH_SIZE = 16
EPOCHS = 5
EMBED_DIM = 128
HIDDEN_DIM = 128

class SentimentDataset(Dataset):
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

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        return self.fc(hidden[-1])

def main():
    df = pd.read_excel("data/zoom_classification_dataset_final.xlsx")
    df = df[["User Review", "Positive/Negative"]].dropna()
    df["User Review"] = df["User Review"].apply(clean_text)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Positive/Negative"])

    X_train, X_val, y_train, y_val = train_test_split(df["User Review"], df["label"], test_size=0.2, random_state=42)

    all_tokens = [tok for text in X_train for tok in text.split()]
    vocab = {w: i+2 for i, (w, _) in enumerate(Counter(all_tokens).most_common())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    train_ds = SentimentDataset(X_train.tolist(), y_train.tolist(), vocab)
    val_ds = SentimentDataset(X_val.tolist(), y_val.tolist(), vocab)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} completed.")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("\U0001F4CA GRU Sentiment Model Performance:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    torch.save(model.state_dict(), "models/legacy_models/gru_sentiment.pth")

if __name__ == "__main__":
    main()
