import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
adv_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset1_alone\king_penguin"
# adv_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset1\FGSM1_epsilon_0.300\king_penguin"

# Load MobileNetV2 pretrained
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
model.eval()

# ImageNet normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Denormalize for visualization & processing in pixel domain
def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return t * std + mean

def norm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mean) / std

# Low-pass filter function in numpy for one image tensor (C,H,W) in [0,1]
def low_pass_filter(img_tensor, radius=30):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # H,W,C

    filtered = np.zeros_like(img)
    rows, cols = img.shape[:2]
    crow, ccol = rows//2, cols//2

    for c in range(3):
        f = np.fft.fft2(img[:, :, c])
        fshift = np.fft.fftshift(f)

        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

        fshift_filtered = fshift * mask

        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        filtered[:, :, c] = img_back

    # Clip to [0,1]
    filtered = np.clip(filtered, 0, 1)
    # Back to tensor C,H,W
    filtered_tensor = torch.tensor(filtered).permute(2, 0, 1)
    return filtered_tensor

# Load images and labels from folder, return list of (original_tensor, adv_tensor)
def load_images(adv_folder):
    original_images = []
    adv_images = []
    file_names = []

    for file in sorted(os.listdir(adv_folder)):
        if file.endswith("_adv.png"):
            adv_path = os.path.join(adv_folder, file)
            orig_path = os.path.join(adv_folder, file.replace("_adv", "_original"))

            adv_img = Image.open(adv_path).convert("RGB")
            # orig_img = Image.open(orig_path).convert("RGB")

            adv_tensor = to_tensor(adv_img)  # [0,1]
            # orig_tensor = to_tensor(orig_img)

            adv_images.append(adv_tensor)
            # original_images.append(orig_tensor)
            file_names.append(file)

    return original_images, adv_images, file_names

# Predict labels using model on normalized tensors
def predict_batch(model, tensors, device):
    preds = []
    with torch.no_grad():
        for img_tensor in tensors:
            img_norm = normalize(img_tensor).unsqueeze(0).to(device)
            output = model(img_norm)
            pred = output.argmax(dim=1).item()
            preds.append(pred)
    return preds

# You need your true labels here to calculate accuracy
# Assuming your dataset_to_imagenet_index['king_penguin'] = 145
true_label = 145

def evaluate(predictions, true_label):
    correct = sum(p == true_label for p in predictions)
    return correct / len(predictions) * 100, correct, len(predictions)

# Main procedure
original_imgs, adv_imgs, _ = load_images(adv_dir)

print(f"Loaded {len(adv_imgs)} adversarial images.")

# Predict on original adv images
adv_preds = predict_batch(model, adv_imgs, device)
acc_adv,correct,total = evaluate(adv_preds, true_label)
print(f"Accuracy on original adversarial images: {acc_adv:.2f}% , {correct}/{total} correct")

# Apply low-pass filter on adversarial images
# Regular 10-step radii
radii_10 = list(range(10, 110, 10))  # [10, 20, ..., 100]

# Additional 5-step radii between 60 and 95
radii_5 = list(range(65, 100, 10))   # [65, 75, 85, 95]

# Combine and sort (remove duplicates just in case)
radii = sorted(set(radii_10 + radii_5))
accuracies = []

for radius in radii:
    filtered_adv_imgs = []
    for img in adv_imgs:
        # Denormalize to pixel domain [0,1]
        img_denorm = denorm(img)
        # Clip just in case
        img_denorm = torch.clamp(img_denorm, 0, 1)
        # Apply low-pass filter with current radius
        filtered_img = low_pass_filter(img_denorm, radius=radius)
        # Normalize back to ImageNet space
        filtered_img_norm = norm(filtered_img)
        filtered_adv_imgs.append(filtered_img_norm)

    # Predict on filtered images
    filtered_preds = predict_batch(model, filtered_adv_imgs, device)
    acc_filtered, correct, total = evaluate(filtered_preds, true_label)
    accuracies.append(acc_filtered)
    print(f"Radius {radius}: Accuracy = {acc_filtered:.2f}% ({correct}/{total})")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(radii, accuracies, marker='o')
plt.title("Accuracy vs Low-pass Filter Radius")
plt.xlabel("Low-pass Filter Radius")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
