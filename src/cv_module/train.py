import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# from cv_module.dataset_loader import load_dataset
from .dataset_loader import load_dataset

from torch.utils.data import DataLoader

MODEL_SAVE_PATH = "models/disease_model/best_model.pth"


# ------------------------------------------------
#   Build EfficientNet-B0 Model
# ------------------------------------------------
def build_model(num_classes: int):
    print(f"\n🔧 Building EfficientNet-B0 for {num_classes} classes...")

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ------------------------------------------------
#   Validation Loop
# ------------------------------------------------
def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


# ------------------------------------------------
#   Training Loop
# ------------------------------------------------
def train_model(
    epochs=10,
    batch_size=32,
    lr=1e-4,
    val_split=0.2,
    reduce_ratio=0.2
):
    print("\n📚 Loading dataset...")
    train_dataset, val_dataset, classes = load_dataset(
        batch_size=batch_size,
        val_split=val_split,
        reduce_ratio=reduce_ratio
    )

    print("\n📦 Dataset loaded:")
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    print("Classes:", classes)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n💻 Using device:", device)

    # Model
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    early_stop_counter = 0
    patience = 3  # You can tweak or disable early stopping

    print("\n🚀 Starting training...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validate after each epoch
        val_loss, val_acc = validate(model, criterion, val_loader, device)

        print(f"\n📊 Epoch {epoch}/{epochs}")
        print(f"Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}\n")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved new BEST model (Acc: {best_acc:.4f})\n")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping condition
        if early_stop_counter >= patience:
            print("⏹ Early stopping triggered!")
            break

    print(f"🎉 Training complete. Best Accuracy: {best_acc:.4f}")
    print(f"📁 Model saved at: {MODEL_SAVE_PATH}")


# ------------------------------------------------
#   Run directly
# ------------------------------------------------
if __name__ == "__main__":
    train_model(
        epochs=10,
        batch_size=32,
        lr=1e-4,
        val_split=0.2,
        reduce_ratio=0.2  # Use 0.05 for faster run, 0.2 for accuracy
    )
