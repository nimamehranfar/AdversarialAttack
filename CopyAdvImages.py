import shutil
import os

# Temporary folder to hold only adversarial images
adv_base_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset2\FGSM2_epsilon_0.300\king_penguin"
adv_temp_eval_dir = r"C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Attacked_Dataset2_alone\king_penguin"
os.makedirs(adv_temp_eval_dir, exist_ok=True)

# Clear the directory first (optional safety)
for f in os.listdir(adv_temp_eval_dir):
    os.remove(os.path.join(adv_temp_eval_dir, f))

# Copy only adversarial images
for f in os.listdir(adv_base_dir):
    if f.endswith("_adv.png"):
        src_path = os.path.join(adv_base_dir, f)
        dst_path = os.path.join(adv_temp_eval_dir, f)
        shutil.copyfile(src_path, dst_path)
