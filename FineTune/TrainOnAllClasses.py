import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import time
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg', 'Agg', etc.
import matplotlib.pyplot as plt


# Base dataset directory
base_dir = "Dataset"

# Mapping class folder names to ImageNet class indices
dataset_to_imagenet_idx = {
    'African elephant': 386,
    'brown bear': 294,
    'chameleon': 47,
    'dragonfly': 319,
    'giant panda': 388,
    'gorilla': 366,
    'king penguin': 145,
    'king_penguin': 145,
    'koala': 105,
    'ladybug': 301,
    'lion': 291,
    'meerkat': 299,
    'orangutan': 365,
    'red fox': 277,
    'snail': 113,
    'tiger': 292,
    'kite': 21,
    'Virginia deer': 352
}

# Transformations for training and validation/test
input_size = 224
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),     # Random crop and resize
    transforms.RandomHorizontalFlip(),                       # Horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2,     # Slight color perturbation
                           saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),                           # Small random rotations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class MultiClassDataset(Dataset):
    def __init__(self, base_dir, dataset_to_imagenet_idx, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.dataset_to_imagenet_idx = dataset_to_imagenet_idx

        # Walk through base_dir recursively
        for root, dirs, files in os.walk(base_dir):
            # Check if the current folder corresponds to a class we care about
            # Folder name will be the last part of root path
            folder_name = os.path.basename(root)
            if folder_name in dataset_to_imagenet_idx:
                label = dataset_to_imagenet_idx[folder_name]
                # Collect image files
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        self.images.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def train_epoch(device, model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total * 100

def evaluate(device, model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total * 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = MultiClassDataset(base_dir, dataset_to_imagenet_idx, transform=transform_train)
    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
    val_dataset.dataset.transform = transform_val_test
    test_dataset.dataset.transform = transform_val_test

    print(f"Dataset sizes: train={train_len}, val={val_len}, test={test_len}")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=14)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=14)

    # Load MobileNetV2 pretrained model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initial evaluation
    val_loss, val_acc = evaluate(device, model, val_loader, criterion)
    print(f"Initial validation accuracy (before training): {val_acc:.2f}%")

    num_epochs = 10
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    epoch_times = []

    print("\nStarting training...\n")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(device, model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(device, model, val_loader, criterion)

        scheduler.step()
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds")

    # Final test evaluation
    test_loss, test_acc = evaluate(device, model, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), "mobilenet_finetuned_multiclass.pth")
    print("Model saved as mobilenet_finetuned_multiclass.pth")

    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Plot loss curves (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == '__main__':
    main()
