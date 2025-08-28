import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch

# ✅ Force MLflow to use local file storage
mlflow.set_tracking_uri("file:///D:/git_hub/mlruns")

# -----------------------------
# 1. Dataset
# -----------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab[token] for token in tokens[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx])

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

# -----------------------------
# 2. Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# -----------------------------
# 3. Train / Evaluate
# -----------------------------
def train_fn(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        acc = (preds.argmax(1) == y).float().mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_fn(model, iterator, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            acc = (preds.argmax(1) == y).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# -----------------------------
# 4. Main
# -----------------------------
def main():
    # Fake data (replace with real IMDB later)
    texts = ["I loved the movie", "I hated the film", "Best movie ever", "Worst movie ever"]
    labels = [1, 0, 1, 0]

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(texts, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    dataset = IMDBDataset(texts, labels, vocab, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab), 100, 128, 2).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # ✅ Track with MLflow
    with mlflow.start_run():
        for epoch in range(2):
            train_loss, train_acc = train_fn(model, loader, optimizer, criterion, device)
            val_loss, val_acc = eval_fn(model, loader, criterion, device)
            print(f"Epoch {epoch+1}/2 | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        # Save model + artifacts
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), "outputs/model.pth")
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts("outputs/")

if __name__ == "__main__":
    main()
