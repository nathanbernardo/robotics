import torchvision.transforms as transforms
import uuid
import torch
import os
import shutil
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.io import read_image, write_png
from pathlib import Path
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import track
from typing import List, Tuple, Dict
from rich.progress import Progress, TaskID

console = Console()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Program to generate augmentated image dataset")
    parser.add_argument("--path", type=Path, help="Path to the image dataset to be augmented")
    parser.add_argument("--output", type=Path, required=True, help="Output directory to store augmented images")
    parser.add_argument("--num_augmentations", type=int, default=50, help="Number of agumentations per image")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test set")

    return parser.parse_args()

def create_augmentation_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ])

def create_albumentations_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.95, 1.05), translate_percent=(0.05, 0.05)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        A.ToGray(p=0.1),
    ])

def generate_unique_filename(original_filename: str, counter: int) -> str:
    name, ext = os.path.splitext(original_filename)
    return f"{name}_aug_{counter:03d}_{uuid.uuid4().hex[:8]}{ext}"

def augment_keurig_image(input_dir: Path, output_dir: Path, num_images: int) -> Dict[str, List[torch.Tensor]]:
    # Define augmentation transforms
    transform = create_augmentation_transform()

    # Get all image files from the input directory
    image_files = get_image_files(input_dir)
    augmented_images = {}

    with Progress() as progress:
        task = progress.add_task("[yellow]Augmenting images...", total=len(image_files) * (num_images + 1))
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            image = read_image(input_path)
            augmented_images[image_file] = [(image_file, image)]

            progress.update(task, advance=1)

            for i in range(num_images):
                augmented_image = transform(image)
                unique_filename = generate_unique_filename(image_file, i+1)
                augmented_images[image_file].append((unique_filename, augmented_image))
                progress.update(task, advance=1)

    console.log(f"[bold green]Successfully generated {num_images} augmented images.  Images saved in {output_dir}[/bold green]")
    return augmented_images

def get_image_files(directory: Path) -> List[str]:
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def split_dataset(augmented_images: Dict[str, List[torch.Tensor]], output_base_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float):
    console.log("[bold]Splitting datasets...[/bold]")

    # Get all images
    all_images = [(filename, img) for img_list in augmented_images.values() for filename, img in img_list]

    # Split the dataset
    train_data, test_val_data = train_test_split(all_images, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(test_val_data, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # Move files to respective directories
    with Progress() as progress:
        train_task = progress.add_task("[yellow]Saving train data", total=len(train_data))
        val_task = progress.add_task("[yellow]Saving validation data", total=len(val_data))
        test_task = progress.add_task("[yellow]Saving test data", total=len(test_data))

        for data, split, task in zip([train_data, val_data, test_data], ['train', 'val', 'test'], [train_task, val_task, test_task]):
            split_dir = output_base_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for filename, img in data:
                output_path = split_dir / filename
                write_png(img, str(output_path))
                progress.update(task, advance=1)

    console.log(f"[bold green]Successfully split datasets. Images in train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}[/bold green]")

def main():
    args = parse_arguments()

    augmented_images = augment_keurig_image(args.path, args.output, args.num_augmentations)

    # Split the agumented dataset
    split_dataset(augmented_images, args.output, args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == '__main__':
    main()
