import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

def load_imdb_data(batch_size, device):
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    # Build vocab
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Reset train iterator
    train_iter = IMDB(split='train')

    def process_data(data_iter):
        texts, labels = [], []
        for label, text in data_iter:
            tensor = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
            texts.append(tensor)
            labels.append(torch.tensor(1 if label == 'pos' else 0, dtype=torch.float))
        return texts, labels

    train_texts, train_labels = process_data(train_iter)
    test_texts, test_labels = process_data(test_iter)

    # Split train into train + val
    split = int(0.8 * len(train_texts))
    train_dataset = list(zip(train_texts[:split], train_labels[:split]))
    val_dataset = list(zip(train_texts[split:], train_labels[split:]))
    test_dataset = list(zip(test_texts, test_labels))

    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts = pad_sequence(texts, batch_first=True)
        labels = torch.stack(labels)
        return texts.to(device), labels.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Wrap vocab in TEXT class
    class TEXT:
        vocab = vocab

    class LABEL:
        pass

    return TEXT, LABEL, train_loader, val_loader, test_loader
