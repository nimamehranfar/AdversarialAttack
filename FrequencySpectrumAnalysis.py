import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms


# Prepare image loader and transformer
to_tensor = transforms.ToTensor()

def load_image(path):
    return to_tensor(Image.open(path).convert('RGB')).numpy()  # shape: (C, H, W), values in [0,1]

def rgb_to_gray(img):
    # Simple luminance conversion
    return 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]

def fft_magnitude(image):
    # Compute 2D FFT and magnitude spectrum of grayscale image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

orig_perturbation_spectra = []
adv_perturbation_spectra = []
perturbation_spectra = []

# Loop over all adversarial images and compute perturbations
for fgsm_type in ["FGSM1", "FGSM2"]:

    # PATHS
    if fgsm_type == "FGSM1":
        base_dir = "Attacked_Dataset1/FGSM1_epsilon_0.300/king_penguin"
    else:
        base_dir = "Attacked_Dataset2/FGSM2_epsilon_0.100/king_penguin"

    for filename in sorted(os.listdir(base_dir)):
        if filename.endswith("_adv.png"):
            adv_path = os.path.join(base_dir, filename)
            orig_path = os.path.join(base_dir, filename.replace("_adv", "_original"))

            adv_img = load_image(adv_path)
            orig_img = load_image(orig_path)

            perturbation = adv_img - orig_img  # shape (3, H, W)


            # Convert to grayscale for frequency analysis
            orig_gray = rgb_to_gray(orig_img)
            adv_gray = rgb_to_gray(adv_img)
            perturb_gray = rgb_to_gray(perturbation)

            # FFT magnitude
            orig_mag_spec = fft_magnitude(orig_gray)
            orig_perturbation_spectra.append(orig_mag_spec)
            adv_mag_spec = fft_magnitude(adv_gray)
            adv_perturbation_spectra.append(adv_mag_spec)

            mag_spec = fft_magnitude(perturb_gray)
            perturbation_spectra.append(mag_spec)

    # Average magnitude spectrum over all perturbations
    orig_avg_spectrum = np.mean(orig_perturbation_spectra, axis=0)
    adv_avg_spectrum = np.mean(adv_perturbation_spectra, axis=0)
    avg_spectrum = np.mean(perturbation_spectra, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Original perturbation spectrum average
    im0 = axs[0].imshow(orig_avg_spectrum, cmap='inferno')
    axs[0].set_title('Average Spectrum of Original Images')
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Adversarial perturbation spectrum average
    im1 = axs[1].imshow(adv_avg_spectrum, cmap='inferno')
    axs[1].set_title('Average Spectrum of Adversarial Images')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Average perturbation spectrum
    im2 = axs[2].imshow(avg_spectrum, cmap='inferno')
    axs[2].set_title('Average Spectrum of Perturbations')
    axs[2].axis('off')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()