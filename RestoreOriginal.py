import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# ---------- CONFIG PATHS ----------
orig_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Original_Dataset_kp\king_penguin"
adv_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset2_alone\king_penguin"
restored_dir_avg = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Restored_Images2_avg\king_penguin"
restored_dir_med = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Restored_Images2_med\king_penguin"

fingerprint_avg_path = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Averages2\avg_perturbation.pt"
fingerprint_med_path = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Averages2\med_perturbation.pt"

os.makedirs(restored_dir_avg, exist_ok=True)
os.makedirs(restored_dir_med, exist_ok=True)

# ---------- LOAD MODEL ----------
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
model.eval()

# ---------- NORMALIZATION ----------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return t * std + mean

def norm(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std

# ---------- RESTORE IMAGE ----------
def restore_image(adv_tensor, fingerprint_tensor):
    return torch.clamp(adv_tensor - fingerprint_tensor, -2, 2)

# ---------- PREDICT FUNCTION ----------
def predict_batch(model, tensors, device):
    preds = []
    with torch.no_grad():
        for img_tensor in tensors:
            img_norm = normalize(img_tensor).unsqueeze(0).to(device)
            output = model(img_norm)
            pred = output.argmax(dim=1).item()
            preds.append(pred)
    return preds

# ---------- EVALUATION ----------
def evaluate(predictions, true_label):
    correct = sum(p == true_label for p in predictions)
    return correct / len(predictions) * 100, correct, len(predictions)

# ---------- LOAD IMAGES FROM FOLDER ----------
def load_images(folder, suffix="", ext=".png"):
    images = []
    filenames = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(suffix + ext):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("RGB")
            img_tensor = to_tensor(img)
            images.append(img_tensor)
            filenames.append(file)
    return images, filenames

# ---------- MAIN ----------
true_label = 145

# 1. Load images
orig_images, orig_filenames = load_images(orig_dir, ext=".jpg")
adv_images, adv_filenames = load_images(adv_dir, suffix="_adv", ext=".png")

print(f"Loaded {len(orig_images)} original, {len(adv_images)} adversarial images")

# 2. Predict
orig_preds = predict_batch(model, orig_images, device)
adv_preds = predict_batch(model, adv_images, device)

# 3. Evaluate
acc_orig, c_orig, t_orig = evaluate(orig_preds, true_label)
acc_adv, c_adv, t_adv = evaluate(adv_preds, true_label)

print(f"Original images accuracy:     {acc_orig:.2f}% ({c_orig}/{t_orig})")
print(f"Adversarial images accuracy:  {acc_adv:.2f}% ({c_adv}/{t_adv})")

# 4. Restore and save images
fp_avg = torch.load(fingerprint_avg_path)
fp_med = torch.load(fingerprint_med_path)

restored_avg = []
restored_med = []

for img_tensor, filename in zip(adv_images, adv_filenames):
    r_avg = restore_image(img_tensor, fp_avg)
    r_avg = denorm(r_avg).clamp(0, 1)
    restored_avg.append(norm(r_avg))
    save_image(r_avg, os.path.join(restored_dir_avg, filename.replace("_adv", "_avg_restored")))

    r_med = restore_image(img_tensor, fp_med)
    r_med = denorm(r_med).clamp(0, 1)
    restored_med.append(norm(r_med))
    save_image(r_med, os.path.join(restored_dir_med, filename.replace("_adv", "_med_restored")))

# 5. Predict on restored images
avg_preds = predict_batch(model, restored_avg, device)
med_preds = predict_batch(model, restored_med, device)

acc_avg, c_avg, t_avg = evaluate(avg_preds, true_label)
acc_med, c_med, t_med = evaluate(med_preds, true_label)

print(f"Restored (avg) accuracy:      {acc_avg:.2f}% ({c_avg}/{t_avg})")
print(f"Restored (med) accuracy:      {acc_med:.2f}% ({c_med}/{t_med})")
