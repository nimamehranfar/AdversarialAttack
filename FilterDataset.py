import os
import shutil
import random

# Path to the original dataset
dataset_path = 'NaturalImageNet'

# Path to save filtered images
filtered_path = 'Filtered_Dataset'

# Number of images per class
images_per_class = 300

# Create filtered_path if not exist
os.makedirs(filtered_path, exist_ok=True)

# List all classes (folders)
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

print(f"Found {len(classes)} classes.")

for cls in classes:
    class_folder = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]

    # Shuffle images randomly
    random.shuffle(images)

    # Take only 200 images (or less if not enough)
    selected_images = images[:images_per_class]

    # Create class folder in filtered_path
    filtered_class_folder = os.path.join(filtered_path, cls)
    os.makedirs(filtered_class_folder, exist_ok=True)

    # Copy selected images
    for img in selected_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(filtered_class_folder, img)
        shutil.copy2(src, dst)

    print(f"Copied {len(selected_images)} images from class '{cls}'.")

print("Filtering complete.")
