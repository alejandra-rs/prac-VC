import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import LambdaLR


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class GPUBatchLoader:
    def __init__(self, inputs, labels, batch_size, train=False, seed=0):
        self.inputs = torch.from_numpy(inputs).cuda()
        self.labels = torch.from_numpy(labels).cuda()
        self.batch_size = batch_size
        self.train = train
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        full, rest = divmod(len(self.labels), self.batch_size)
        return full if self.train or rest == 0 else full + 1

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, generator=self.generator) if self.train else torch.arange(n)
        order = order.cuda()
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            batch = order[start:start + self.batch_size]
            yield self.inputs[batch], self.labels[batch]


def warmup(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    return factor


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_transform, eval_transform, checkpoint, patience,
                 writer=None, on_epoch_start=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.checkpoint = checkpoint
        self.patience = patience
        self.writer = writer
        self.on_epoch_start = on_epoch_start
        self.scaler = torch.amp.GradScaler("cuda")

    def train_epoch(self, loader, scheduler):
        self.model.train()
        total_loss, correct, n = 0, 0, 0
        for x, y in loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = self.model(self.train_transform(x))
                loss = self.loss_fn(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        return total_loss / n, correct / n

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, predictions, labels = 0, [], []
        for x, y in loader:
            with torch.autocast("cuda"):
                logits = self.model(self.eval_transform(x))
            total_loss += self.loss_fn(logits.float(), y).item() * len(y)
            predictions.append(logits.argmax(1))
            labels.append(y)
        predictions, labels = torch.cat(predictions).cpu().numpy(), torch.cat(labels).cpu().numpy()
        return total_loss / len(labels), (predictions == labels).mean(), f1_score(labels, predictions, average="macro")

    def fit(self, train_loader, val_loader, epochs, warmup_epochs):
        steps = len(train_loader)
        scheduler = LambdaLR(self.optimizer, warmup(epochs * steps, warmup_epochs * steps))
        best_f1, bad_epochs = -1, 0
        for epoch in range(1, epochs + 1):
            if self.on_epoch_start:
                self.on_epoch_start(epoch)
            train_loss, train_acc = self.train_epoch(train_loader, scheduler)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            if self.writer:
                self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
                self.writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
                self.writer.add_scalar("val_f1_macro", val_f1, epoch)
            if val_f1 > best_f1:
                best_f1, bad_epochs = val_f1, 0
                torch.save(self.model.state_dict(), self.checkpoint)
            else:
                bad_epochs += 1
                if bad_epochs == self.patience:
                    break
        if self.writer:
            self.writer.close()
        self.model.load_state_dict(torch.load(self.checkpoint))
        return self.model
