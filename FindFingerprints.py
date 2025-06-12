import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg', 'Agg', etc.
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Define paths
base_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset1\FGSM1_epsilon_0.300\king_penguin"
output_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Averages"
os.makedirs(output_dir, exist_ok=True)

# ----- TRANSFORMS -----
to_tensor = transforms.ToTensor()

# Denormalization transform for ImageNet
def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

# Normalize to [0, 1] for visualization only
def normalize_for_display(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)

# ----- LOAD IMAGES -----
adv_images = []
orig_images = []

for file in sorted(os.listdir(base_dir)):
    if file.endswith("_adv.png"):
        adv_path = os.path.join(base_dir, file)
        orig_path = os.path.join(base_dir, file.replace("_adv", "_original"))

        adv_img = to_tensor(Image.open(adv_path).convert("RGB"))
        orig_img = to_tensor(Image.open(orig_path).convert("RGB"))

        # Denormalize if these were normalized tensors
        adv_images.append(denorm(adv_img))
        orig_images.append(denorm(orig_img))

# ----- STACK AND COMPUTE -----
adv_stack = torch.stack(adv_images)
orig_stack = torch.stack(orig_images)

avg_adv = torch.mean(adv_stack, dim=0)
med_adv = torch.median(adv_stack, dim=0).values

avg_orig = torch.mean(orig_stack, dim=0)
med_orig = torch.median(orig_stack, dim=0).values

perturbation_avg = avg_adv - avg_orig
perturbation_med = med_adv - med_orig

# ----- SAVE RESULTS -----
save_image(normalize_for_display(avg_adv), os.path.join(output_dir, "avg_adversarial.png"))
save_image(normalize_for_display(med_adv), os.path.join(output_dir, "median_adversarial.png"))
save_image(normalize_for_display(avg_orig), os.path.join(output_dir, "avg_original.png"))
save_image(normalize_for_display(med_orig), os.path.join(output_dir, "median_original.png"))
save_image(normalize_for_display(perturbation_avg), os.path.join(output_dir, "avg_perturbation.png"))
save_image(normalize_for_display(perturbation_med), os.path.join(output_dir, "median_perturbation.png"))

torch.save(perturbation_avg, r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Averages\avg_perturbation.pt")
torch.save(perturbation_med, r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Averages\med_perturbation.pt")

# ----- OPTIONAL: DISPLAY -----
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].imshow(normalize_for_display(avg_orig).permute(1, 2, 0))
axs[0, 0].set_title("Average Original")

axs[0, 1].imshow(normalize_for_display(avg_adv).permute(1, 2, 0))
axs[0, 1].set_title("Average Adversarial")

axs[0, 2].imshow(normalize_for_display(perturbation_avg).permute(1, 2, 0))
axs[0, 2].set_title("Average Perturbation")

axs[1, 0].imshow(normalize_for_display(med_orig).permute(1, 2, 0))
axs[1, 0].set_title("Median Original")

axs[1, 1].imshow(normalize_for_display(med_adv).permute(1, 2, 0))
axs[1, 1].set_title("Median Adversarial")

axs[1, 2].imshow(normalize_for_display(perturbation_med).permute(1, 2, 0))
axs[1, 2].set_title("Median Perturbation")

for ax in axs.flatten():
    ax.axis("off")

plt.tight_layout()
plt.show()