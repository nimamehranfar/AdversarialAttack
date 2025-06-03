import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet50, ResNet50_Weights
from collections import Counter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = r'C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Filtered_Dataset'

dataset_to_imagenet_idx = {
    'African elephant': 386,
    'brown bear': 294,
    'chameleon': 47,
    'dragonfly': 319,
    'giant panda': 388,
    'gorilla': 366,
    'king penguin': 145,
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

# Reverse dict for idx to class name
imagenet_idx_to_class = {v: k for k, v in dataset_to_imagenet_idx.items()}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        folder_class_name = self.classes[class_idx]
        target = dataset_to_imagenet_idx[folder_class_name]
        return sample, target, index  # return index for error tracking

dataset = CustomImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
mobilenet.eval()

def detailed_evaluation(model, dataloader):
    per_class_correct = {cls: 0 for cls in dataset_to_imagenet_idx.values()}
    per_class_total = {cls: 0 for cls in dataset_to_imagenet_idx.values()}
    wrong_preds = {cls: [] for cls in dataset_to_imagenet_idx.values()}

    with torch.no_grad():
        for images, targets, indices in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            for i in range(len(targets)):
                true_label = targets[i].item()
                pred_label = preds[i].item()
                per_class_total[true_label] += 1
                if pred_label == true_label:
                    per_class_correct[true_label] += 1
                else:
                    # Store wrong prediction info: (image index in dataset, predicted class)
                    wrong_preds[true_label].append((indices[i].item(), pred_label))

    # Calculate accuracy per class
    per_class_acc = {}
    for cls in per_class_total:
        total = per_class_total[cls]
        correct = per_class_correct[cls]
        acc = correct / total if total > 0 else 0
        per_class_acc[cls] = acc

    return per_class_acc, per_class_correct, per_class_total, wrong_preds

per_class_acc, per_class_correct, per_class_total, wrong_preds = detailed_evaluation(mobilenet, dataloader)

print("Per-class accuracy and stats:")
for imagenet_idx, acc in per_class_acc.items():
    cls_name = imagenet_idx_to_class[imagenet_idx]
    print(f"{cls_name}: {per_class_correct[imagenet_idx]}/{per_class_total[imagenet_idx]} correct - Accuracy: {acc*100:.2f}%")

# Find the class with highest accuracy
best_class_idx = max(per_class_acc, key=per_class_acc.get)
best_class_name = imagenet_idx_to_class[best_class_idx]
print(f"\nClass with highest accuracy: {best_class_name} ({per_class_acc[best_class_idx]*100:.2f}%)")

# Show wrong predictions for this class
print(f"\nWrong predictions for class '{best_class_name}':")
if wrong_preds[best_class_idx]:
    for idx, pred_idx in wrong_preds[best_class_idx]:
        pred_class_name = imagenet_idx_to_class.get(pred_idx, f"ImageNet class {pred_idx}")
        print(f"  Dataset image index {idx} predicted as {pred_class_name}")
else:
    print("  None, perfect classification!")
