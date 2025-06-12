import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet50, ResNet50_Weights
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg', 'Agg', etc.
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} as device")

dataset_path = r'C:\Users\mehra\IdeaProjects\AdversarialAttackProject\Filtered_Dataset'

dataset_to_imagenet_index = {
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

imagenet_index_to_name = {v: k.replace(" ", "_") for k, v in dataset_to_imagenet_index.items()}

king_penguin_index = 145

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, class_index = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        folder_class_name = self.classes[class_index]
        target = dataset_to_imagenet_index[folder_class_name]
        return sample, target, index


dataset = CustomImageFolder(root=dataset_path, transform=transform)

# Filter only king penguin images
king_penguin_subset = [i for i, (_, label, _) in enumerate(dataset) if label == king_penguin_index]
king_penguin_loader = DataLoader(torch.utils.data.Subset(dataset, king_penguin_subset), batch_size=1, shuffle=False)
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)

def denorm(data):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(data.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(data.device)
    return data * std + mean

def fgsm_attack(image, epsilon, data_grad):
    # Get sign of gradients
    sign_data_grad = data_grad.sign()
    # Add perturbation
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp to [0, 1]
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def attack_king_penguins(model, device, test_loader, epsilon, output_folder=0):
    model.eval()
    success, total = 0, 0

    # Inverse normalization for saving images
    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    for i, (image, label, index) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)
        image.requires_grad = True

        output = model(image)
        init_pred = output.max(1, keepdim=True)[1]

        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        perturbed_image = fgsm_attack(image, epsilon, data_grad)

        # Save original and adversarial images
        orig_img = inv_norm(image.squeeze().cpu()).clamp(0, 1)
        adv_img = inv_norm(perturbed_image.squeeze().detach().cpu()).clamp(0, 1)

        if output_folder !=0:
            # os.makedirs(output_folder, exist_ok=True)
            label_index = label.item()
            class_name = imagenet_index_to_name.get(label_index, "unknown_class")
            output_folder_epsilon = os.path.join(output_folder, f"FGSM1_epsilon_{epsilon:.3f}")
            # output_folder_epsilon = output_folder + f"_epsilon_{epsilon:.3f}"
            class_folder = os.path.join(output_folder_epsilon, class_name)
            os.makedirs(class_folder, exist_ok=True)
            torchvision.utils.save_image(orig_img, f"{class_folder}/{i:03d}_original.png")
            torchvision.utils.save_image(adv_img, f"{class_folder}/{i:03d}_adv.png")

        total += 1
        if init_pred.item() != label.item():
            continue  # Skip already misclassified

        output_adv = model(perturbed_image)
        final_pred = output_adv.max(1, keepdim=True)[1]

        # if final_pred.item() != label.item():
        #     success += 1

        if final_pred.item() == label.item():
            success += 1

    success_rate = (success / total) * 100 if total > 0 else 0.0
    print(f"\nModel Accuracy After FGSM Attack on King Penguins (epsilon={epsilon}): {success_rate:.2f}%")
    return success_rate

#success_rate = attack_king_penguins(model, device, king_penguin_loader, 0.3, "attacked_king_penguin")


def attack_with_denorm_and_re_norm(model, device, test_loader, epsilon, output_folder=0):
    model.eval()
    correct = 0
    total = 0
    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    for i, (data, target, index) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        # Collect "datagrad"
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data.squeeze())

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad.squeeze())

        # Reapply normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        perturbed_data_normalized = normalize(perturbed_data).unsqueeze(0)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]

        orig_img = inv_norm(data.squeeze().cpu()).clamp(0, 1)
        adv_img = inv_norm(perturbed_data_normalized.squeeze().detach().cpu()).clamp(0, 1)

        if output_folder !=0:
            # os.makedirs(output_folder, exist_ok=True)
            label_index = target.item()
            class_name = imagenet_index_to_name.get(label_index, "unknown_class")
            output_folder_epsilon = os.path.join(output_folder, f"FGSM2_epsilon_{epsilon:.3f}")
            # output_folder_epsilon = output_folder + f"_epsilon_{epsilon:.3f}"
            class_folder = os.path.join(output_folder_epsilon, class_name)
            os.makedirs(class_folder, exist_ok=True)
            torchvision.utils.save_image(orig_img, f"{class_folder}/{i:03d}_original.png")
            torchvision.utils.save_image(adv_img, f"{class_folder}/{i:03d}_adv.png")


        total += 1
        if init_pred.item() != target.item():
            continue  # skip if already misclassified

        # if final_pred.item() != target.item():
        #     correct += 1

        if final_pred.item() == target.item():
            correct += 1


    success_rate = (correct / total) * 100 if total > 0 else 0.0
    print(f"Model Accuracy After FGSM Attack2 on King Penguins (epsilon={epsilon}): {success_rate:.2f}%")
    return success_rate


rate1 = attack_king_penguins(model, device, king_penguin_loader, 0.3, 0)
rate2 = attack_with_denorm_and_re_norm(model, device, king_penguin_loader, 0.3, 0)

epsilons = np.linspace(0, 0.3, 10)
success_rates1 = []
success_rates2 = []

for epsilon in epsilons:
    # rate1 = attack_king_penguins(model, device, king_penguin_loader, epsilon, 0)
    # rate2 = attack_with_denorm_and_re_norm(model, device, king_penguin_loader, epsilon, 0)
    # rate1 = attack_king_penguins(model, device, king_penguin_loader, epsilon, "Attacked_Dataset1")
    # rate2 = attack_with_denorm_and_re_norm(model, device, king_penguin_loader, epsilon, "Attacked_Dataset2")
    success_rates1.append(rate1)
    success_rates2.append(rate2)

plt.figure(figsize=(8,5))
plt.plot(epsilons, success_rates1, marker='o', mfc="white", label='FGSM1 (No Denorm)')
plt.plot(epsilons, success_rates2, marker='s', mfc="black", label='FGSM2 (With Denorm/Re-norm)')
plt.title("Model Accuracy After FGSM Attack on King Penguins vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Model Accuracy (%)")
plt.grid(True)
plt.show()
