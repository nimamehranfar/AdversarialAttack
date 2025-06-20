import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.utils import save_image
import numpy as np
import cv2

# -------- CONFIG --------
dataset_path = '../Filtered_Dataset'
adv_dataset_dir = 'Dataset/Adversarial_Dataset'
fingerprints_dir = 'Dataset/Fingerprints'
restored_dir = 'Dataset/Restored'
denoised_dir = 'Dataset/Denoised'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------- Image normalization for MobileNetV2 --------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(t.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(t.device)
    return t * std + mean

def norm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(t.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(t.device)
    return (t - mean) / std

# -------- FGSM Attack Implementations --------
def fgsm1_attack(model, images, epsilon=0.3):
    images = images.to(device)
    images.requires_grad = True

    outputs = model(normalize(images))
    loss = torch.nn.functional.cross_entropy(outputs, torch.zeros(len(images), dtype=torch.long).to(device))  # dummy zero labels, no eval here
    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.sign()
    adv_images = images + epsilon * grad_sign
    # adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()

def fgsm2_attack(model, images, epsilon=0.1):
    images = images.to(device)
    images_denorm = denorm(images)
    images_denorm.requires_grad = True

    outputs = model(normalize(images_denorm))
    loss = torch.nn.functional.cross_entropy(outputs, torch.zeros(len(images), dtype=torch.long).to(device))  # dummy labels
    model.zero_grad()
    loss.backward()

    grad_sign = images_denorm.grad.sign()
    adv_denorm = images_denorm + epsilon * grad_sign
    # adv_denorm = torch.clamp(adv_denorm, 0, 1)

    adv_images = norm(adv_denorm)
    return adv_images.detach()

# -------- Helper Functions --------
def load_images_from_folder(folder_path, target_size=(224,224)):
    images = []
    filenames = []
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, file)
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
            filenames.append(file)
    return images, filenames

def save_images(tensors, filenames, output_folder, suffix=""):
    os.makedirs(output_folder, exist_ok=True)
    for tensor, fname in zip(tensors, filenames):
        base_name = os.path.splitext(fname)[0]
        save_path = os.path.join(output_folder, base_name + suffix + ".png")
        save_image(tensor, save_path)

def compute_fingerprints(orig_tensors, adv_tensors):
    orig_stack = torch.stack(orig_tensors).to(device)
    adv_stack = torch.stack(adv_tensors).to(device)
    avg_orig = torch.mean(orig_stack, dim=0)
    med_orig = torch.median(orig_stack, dim=0).values
    avg_adv = torch.mean(adv_stack, dim=0)
    med_adv = torch.median(adv_stack, dim=0).values
    perturb_avg = avg_adv - avg_orig
    perturb_med = med_adv - med_orig
    return perturb_avg.cpu(), perturb_med.cpu()

def restore_images(adv_tensors, fingerprint):
    restored = []
    for adv_img in adv_tensors:
        rest = torch.clamp(adv_img - fingerprint, -2, 2)
        restored.append(denorm(rest))
    return restored

def low_pass_filter(img_tensor, radius=30):
    img = img_tensor.permute(1,2,0).cpu().numpy()  # H,W,C
    filtered = np.zeros_like(img)
    rows, cols = img.shape[:2]
    crow, ccol = rows//2, cols//2
    for c in range(3):
        f = np.fft.fft2(img[:,:,c])
        fshift = np.fft.fftshift(f)
        mask = np.zeros((rows,cols), np.uint8)
        cv2.circle(mask, (ccol,crow), radius, 1, thickness=-1)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        filtered[:,:,c] = img_back
    filtered = np.clip(filtered, 0, 1)
    filtered_tensor = torch.tensor(filtered).permute(2,0,1)
    return filtered_tensor

# -------- MAIN PROCESS --------
def main():
    # Load pretrained MobileNetV2
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
    model.eval()

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Found classes: {classes}")

    for cls in classes:
        print(f"\nProcessing class '{cls}'")
        class_dir = os.path.join(dataset_path, cls)

        # Load original images
        orig_images, orig_filenames = load_images_from_folder(class_dir)
        print(f"Loaded {len(orig_images)} images from class '{cls}'")
        orig_tensors = [norm(img) for img in orig_images]  # normalize for model input

        # Generate adversarial images and save
        for fgsm_type in ["FGSM1", "FGSM2"]:
            adv_out_dir = os.path.join(adv_dataset_dir, fgsm_type, cls)
            os.makedirs(adv_out_dir, exist_ok=True)

            adv_tensors = []
            batch_size = 16
            for i in range(0, len(orig_tensors), batch_size):
                batch = torch.stack(orig_tensors[i:i+batch_size]).to(device)
                if fgsm_type == "FGSM1":
                    adv_batch = fgsm1_attack(model, batch, epsilon=0.3)
                else:
                    adv_batch = fgsm2_attack(model, batch, epsilon=0.1)
                adv_tensors.extend(denorm(adv_batch.cpu()).clamp(0, 1))

            save_images(adv_tensors, orig_filenames, adv_out_dir, suffix="_adv")
            print(f"Saved adversarial images with {fgsm_type} for class '{cls}'")

        for fgsm_type in ["FGSM1", "FGSM2"]:
            # Compute fingerprints from FGSM adversarial images
            adv_fgsm_dir = os.path.join(adv_dataset_dir, fgsm_type, cls)
            adv_images, adv_filenames = load_images_from_folder(adv_fgsm_dir)
            if len(orig_images) != len(adv_images):
                print("Mismatch in number of original and adversarial images! Skipping fingerprint computation for this class.")
                continue

            perturb_avg, perturb_med = compute_fingerprints(orig_tensors, adv_images)

            # Save fingerprints
            fingerprint_cls_dir = os.path.join(fingerprints_dir, fgsm_type, cls)
            os.makedirs(fingerprint_cls_dir, exist_ok=True)
            torch.save(perturb_avg, os.path.join(fingerprint_cls_dir, "avg_perturbation.pt"))
            torch.save(perturb_med, os.path.join(fingerprint_cls_dir, "median_perturbation.pt"))
            print(f"Saved fingerprints (average and median) with {fgsm_type} for class '{cls}'")

            # Restore images by subtracting fingerprints and save
            restored_avg = restore_images(adv_images, perturb_avg)
            restored_med = restore_images(adv_images, perturb_med)

            restored_avg_dir = os.path.join(restored_dir, fgsm_type, "avg", cls)
            restored_med_dir = os.path.join(restored_dir, fgsm_type, "median", cls)
            os.makedirs(restored_avg_dir, exist_ok=True)
            os.makedirs(restored_med_dir, exist_ok=True)

            save_images(restored_avg, adv_filenames, restored_avg_dir, suffix="_restored_avg")
            save_images(restored_med, adv_filenames, restored_med_dir, suffix="_restored_med")
            print(f"Saved restored images (avg and median) with {fgsm_type} for class '{cls}'")

            # Denoise adversarial images with low-pass filter for varying radius and save
            radii = [90]
            for radius in radii:
                denoise_dir = os.path.join(denoised_dir, fgsm_type, f"radius_{radius}", cls)
                os.makedirs(denoise_dir, exist_ok=True)
                denoised_imgs = []
                for adv_img in adv_images:
                    filtered = low_pass_filter(adv_img, radius)
                    denoised_imgs.append(filtered)
                save_images(denoised_imgs, adv_filenames, denoise_dir, suffix=f"_denoised_r{radius}")
            print(f"Denoised adversarial images with {fgsm_type} for class '{cls}' with radii {radii}")

    print("\nProcessing completed for all classes!")

if __name__ == "__main__":
    main()
