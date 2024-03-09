import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def get_accuracy(outputs, labels):

    if outputs.dim() == 2 and outputs.shape[-1] > 1:
        return get_multiclass_accuracy(outputs, labels)
    else:
        y_prob = torch.sigmoid(outputs).view(-1)
        y_prob = y_prob > 0.5
        return (labels == y_prob).sum().item()

def get_multiclass_accuracy(outputs, labels):
    probas = torch.softmax(outputs, dim=-1)
    preds = torch.argmax(probas, dim=-1)
    correct = (preds == labels).sum()
    acc = correct
    return acc


def train_epoch(model, dloader, loss_fn, optimizer, device, classify=True, label_index=0, compute_auc=False):
    with torch.autograd.set_detect_anomaly(True):
        running_loss = 0.0
        n_samples = 0
        all_probas = np.array([])
        all_labels = np.array([])
        if classify:
            running_acc = 0.0
        for i, data in enumerate(dloader):
            if len(data.y.shape) > 1:
                labels = data.y[:, label_index].view(-1, 1).flatten()
                labels = labels.float()
            else:
                labels = data.y.flatten()
            if -1 in labels:
                labels = (labels + 1) / 2
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
                labels = labels.long()

            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if 'train_mask' in data.keys:
                outputs = outputs[data.train_mask]
                labels = labels[data.train_mask]
            n_samples += len(labels)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            if compute_auc:
                probas = torch.softmax(outputs, dim=-1)
                all_probas = np.concatenate((all_probas, probas.detach().cpu().numpy()[:, 1]))
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)

        if compute_auc:
            auc = roc_auc_score(all_labels, all_probas)

        if classify:
            if compute_auc:
                return running_loss / len(dloader), running_acc / n_samples, auc
            else:
                return running_loss / len(dloader), running_acc / n_samples, -1
        else:
            return running_loss / len(dloader), -1


def test_epoch(model, dloader, loss_fn, device, classify=True, label_index=0, compute_auc=False, val_mask=False):
    with torch.no_grad():
        running_loss = 0.0
        all_probas = np.array([])
        all_labels = np.array([])
        n_samples = 0
        if classify:
            running_acc = 0.0
        model.eval()
        for i, data in enumerate(dloader):
            if len(data.y.shape) > 1:
                labels = data.y[:, label_index].view(-1, 1).flatten()
                labels = labels.float()
            else:
                labels = data.y.flatten()
            if -1 in labels:
                labels = (labels + 1) / 2
            if loss_fn.__class__.__name__ == 'CrossEntropyLoss':
                labels = labels.long()
            inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model.forward(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if 'test_mask' in data.keys:
                if val_mask:
                    outputs = outputs[data.val_mask]
                    labels = labels[data.val_mask]
                else:
                    outputs = outputs[data.test_mask]
                    labels = labels[data.test_mask]
            n_samples += len(labels)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            if classify:
                running_acc += get_accuracy(outputs, labels)
            if compute_auc:
                probas = torch.softmax(outputs, dim=-1)
                all_probas = np.concatenate((all_probas, probas.detach().cpu().numpy()[:, 1]))
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))

        if compute_auc:
            auc = roc_auc_score(all_labels, all_probas)
        if classify:
            if compute_auc:
                return running_loss / len(dloader), running_acc / n_samples, auc
            else:
                return running_loss / len(dloader), running_acc / n_samples, -1
        else:
            return running_loss / len(dloader), -1

