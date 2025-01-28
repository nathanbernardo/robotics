import torch
import torchvision.transforms as transforms
import os
import shutil
from torchvision.io import read_image, write_png
from pathlib import Path
from sklearn.model_selection import train_test_split

def augment_keurig_image(input_dir, output_dir, num_images=50):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    # Define augmentation transforms
    augmentation = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ])

    # Get all image files from the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    total_augmented = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        image = read_image(input_path)

        for i in range(num_images):
            augmented_image = augmentation(image)
            output_filename = f"augmented_{os.path.splitext(image_file)[0]}_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            write_png(augmented_image, output_path)
            total_augmented += 1

    print(f"{num_images} augmented images saved in {output_dir}")

def split_dataset(input_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Get all images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Split the dataset
    train_files, test_val_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # Create directories
    for split in ['train', 'val', 'test']:
        Path(os.path.join(output_base_dir, split)).mkdir(parents=True, exist_ok=True)

    # Move files to respective directories
    for files, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        for file in files:
            src = os.path.join(input_dir, file)
            dst = os.path.join(output_base_dir, split, file)

            shutil.copy(src, dst)
    print(f"Dataset split complete. Images in train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

# Usage
input_image_path = "./pre_k_elite_dataset"
output_directory = "./post_k_elite_data"
augment_keurig_image(input_image_path, output_directory)

# Split the agumented dataset
split_dataset(output_directory, "./split_dataset")
