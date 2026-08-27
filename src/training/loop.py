"""The shared training loop, used identically for CV folds and the final
full-development-set training run - only the data passed in differs.
"""
import copy

import torch
from torch.utils.data import DataLoader

from .models import set_stage


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += batch_size
    return total_loss / n, correct / n


def train_with_early_stopping(
    model,
    model_name,
    train_ds,
    val_ds,
    class_weights,
    device,
    stages=(1, 2),
    epochs_per_stage=(15, 15),
    lr_per_stage=(1e-3, 1e-4),
    batch_size=32,
    patience=5,
    seed=0,
    num_workers=2,
):
    """Runs the staged freeze/unfreeze training plan with early stopping on
    validation loss, tracked across the WHOLE run (not reset per stage) so a
    stage transition that doesn't help still lets early stopping kick in.

    Returns (best_state_dict, history) where history is a list of per-epoch
    dicts (epoch, stage, train_loss, train_acc, val_loss, val_acc).
    """
    torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    global_epoch = 0

    for stage, max_epochs, lr in zip(stages, epochs_per_stage, lr_per_stage):
        set_stage(model, model_name, stage)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        # Reset the patience counter (not the best-checkpoint tracking) at
        # each stage transition, so newly-unfrozen parameters get a full
        # patience budget to prove themselves rather than inheriting a
        # near-exhausted counter from the previous stage.
        epochs_without_improvement = 0

        for _ in range(max_epochs):
            global_epoch += 1
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            history.append({
                "epoch": global_epoch,
                "stage": stage,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    return best_state, history
