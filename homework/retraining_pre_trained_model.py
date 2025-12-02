import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics'))
from datasets import CustomImageDataset


def get_model(num_classes, device):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate_on_test(model, test_loader, device, criterion):
    model.eval()
    test_running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    avg_test_loss = test_running_loss / len(test_loader)
    test_acc = 100 * correct_test / total_test
    return avg_test_loss, test_acc


def train_model():
    BATCH_SIZE = 32
    NUM_EPOCHS = 6
    LR = 1e-3
    IMG_SIZE = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    current_dir = Path(__file__).parent
    root_train = current_dir / ".." / "data" / "train"
    root_train = root_train.resolve()

    root_val = current_dir / ".." / "data" / "val"
    root_val = root_val.resolve()

    root_test = current_dir / ".." / "data" / "test"
    root_test = root_test.resolve()

    train_dataset = CustomImageDataset(str(root_train), transform=transform)
    class_names = train_dataset.get_class_names()
    num_classes = len(class_names)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomImageDataset(str(root_val), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = CustomImageDataset(str(root_test), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    test_loss, test_acc = evaluate_on_test(model, test_loader, device, criterion)
    print(f"Итоговый результат: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.2f}%")

    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'go-', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# train_model()