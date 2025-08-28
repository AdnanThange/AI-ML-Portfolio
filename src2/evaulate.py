def evaluate_fn(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / len(dataloader), total_correct / total_count
