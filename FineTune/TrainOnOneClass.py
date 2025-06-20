import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image

# --- Paths ---
base_dir = "Dataset"

# Datasets folders with images of king_penguin (one class)
folders = [
    "Original_Dataset_kp",
    "Attacked_Dataset1_alone",
    "Attacked_Dataset2_alone",
    "Restored_Images1_avg",
    "Restored_Images1_med",
    "Restored_Images2_avg",
    "Restored_Images2_med",
]

# ImageNet class index for king_penguin
king_penguin_index = 145

# --- Transform ---
input_size = 224
transform_train = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
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

class SingleClassDataset(Dataset):
    def __init__(self, folders, base_dir, transform=None):
        self.images = []
        self.transform = transform
        for folder in folders:
            folder_path = os.path.join(base_dir, folder, "king_penguin")
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(folder_path, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = king_penguin_index
        return img, label

def train_epoch(device,model, loader, criterion, optimizer):
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

def evaluate(device,model, loader, criterion):
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
    full_dataset = SingleClassDataset(folders, base_dir, transform=transform_train)

    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    # Use val/test transforms
    val_dataset.dataset.transform = transform_val_test
    test_dataset.dataset.transform = transform_val_test

    print(f"Dataset sizes: train={train_len}, val={val_len}, test={test_len}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(device,model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(device,model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    test_loss, test_acc = evaluate(device,model, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), "mobilenet_finetuned_king_penguin.pth")
    print("Model saved as mobilenet_finetuned_king_penguin.pth")

if __name__ == '__main__':
    main()
