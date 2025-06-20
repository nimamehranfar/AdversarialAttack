import shutil
import os

# Temporary folder to hold only adversarial images
adv_base_dir = "Attacked_Dataset1/FGSM1_epsilon_0.300/king_penguin"
adv_temp_eval_dir = "Attacked_Dataset1_alone/king_penguin"
adv_base_dir2 = "Attacked_Dataset2/FGSM2_epsilon_0.100/king_penguin"
adv_temp_eval_dir2 = "Attacked_Dataset2_alone/king_penguin"

os.makedirs(adv_temp_eval_dir, exist_ok=True)
os.makedirs(adv_temp_eval_dir2, exist_ok=True)

# Clear the directory first (optional safety)
for f in os.listdir(adv_temp_eval_dir):
    os.remove(os.path.join(adv_temp_eval_dir, f))

# Copy only adversarial images
for f in os.listdir(adv_base_dir):
    if f.endswith("_adv.png"):
        src_path = os.path.join(adv_base_dir, f)
        dst_path = os.path.join(adv_temp_eval_dir, f)
        shutil.copyfile(src_path, dst_path)

# Clear the directory first (optional safety)
for f in os.listdir(adv_temp_eval_dir2):
    os.remove(os.path.join(adv_temp_eval_dir2, f))

# Copy only adversarial images
for f in os.listdir(adv_base_dir2):
    if f.endswith("_adv.png"):
        src_path = os.path.join(adv_base_dir2, f)
        dst_path = os.path.join(adv_temp_eval_dir2, f)
        shutil.copyfile(src_path, dst_path)
