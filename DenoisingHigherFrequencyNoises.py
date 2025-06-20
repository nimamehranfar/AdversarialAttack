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
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
def low_pass_filter(img_tensor, radius=90):
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



success_rates1 = []
success_rates2 = []

# Main loop
for fgsm_type in ["FGSM1", "FGSM2"]:

    # PATHS
    if fgsm_type == "FGSM1":
        adv_dir = "Attacked_Dataset1_alone/king_penguin"
        out_dir = "Denoised1/king_penguin"
    else:
        adv_dir = "Attacked_Dataset2_alone/king_penguin"
        out_dir = "Denoised2/king_penguin"

    os.makedirs(out_dir, exist_ok=True)
    original_imgs, adv_imgs, file_names = load_images(adv_dir)

    print(f"Loaded {len(adv_imgs)} adversarial images.")

    # Predict on original adv images
    adv_preds = predict_batch(model, adv_imgs, device)
    acc_adv,correct,total = evaluate(adv_preds, true_label)
    print(f"Accuracy on original adversarial images: {acc_adv:.2f}% , {correct}/{total} correct")

    # Regular 10-step radii
    radii_10 = list(range(10, 120, 10))  # [10, 20, ..., 100]

    # Additional 5-step radii between 60 and 95
    radii_5 = list(range(75, 105, 10))   # [65, 75, 85, 95]

    # Combine and sort (remove duplicates just in case)
    radii = sorted(set(radii_10 + radii_5))
    # accuracies = []

    # Apply low-pass filter on adversarial images
    for radius in radii:
        filtered_adv_imgs = []
        for img,filename in zip(adv_imgs, file_names):
            # Denormalize to pixel domain [0,1]
            img_denorm = denorm(img)
            # Clip just in case
            img_denorm = torch.clamp(img_denorm, 0, 1)
            # Apply low-pass filter with current radius
            filtered_img = low_pass_filter(img_denorm, radius=radius)

            # Save Img on radius 90
            if radius == 90:
                denoised_img = filtered_img.clamp(0, 1)
                save_image(denoised_img, os.path.join(out_dir, filename.replace("_adv", "_denoised")))

            # Normalize back to ImageNet space
            filtered_img_norm = norm(filtered_img)
            filtered_adv_imgs.append(filtered_img_norm)

        # Predict on filtered images
        filtered_preds = predict_batch(model, filtered_adv_imgs, device)
        acc_filtered, correct, total = evaluate(filtered_preds, true_label)
        # accuracies.append(acc_filtered)
        if fgsm_type == "FGSM1":
            success_rates1.append(acc_filtered)
        else:
            success_rates2.append(acc_filtered)

        print(f"Radius {radius}: Accuracy = {acc_filtered:.2f}% ({correct}/{total})")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(radii, success_rates1, marker='o', mfc="white", label='FGSM1 (No Denorm)')
plt.plot(radii, success_rates2, marker='s', mfc="black", label='FGSM2 (With Denorm/Re-norm)')
plt.title("Model Accuracy on Both Denoised FGSM vs Low-pass Filter Radius")
plt.xlabel("Low-pass Filter Radius")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
