import typer
import uuid
import torch
import os
import argparse
import albumentations as A
from torchvision.io import read_image, write_png
from pathlib import Path
from sklearn.model_selection import train_test_split
from rich.console import Console
from typing import List, Dict
from rich.progress import Progress

console = Console()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Program to generate augmentated image dataset"
    )
    parser.add_argument(
        "--path", type=Path, help="Path to the image dataset to be augmented"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory to store augmented images",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=50,
        help="(Default: 50). Number of agumentations per image.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="(Default: 0.7). Ratio of training set.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="(Default: 0.15). Ratio of validation set.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="(Default: 0.15). Ratio of test set.",
    )

    return parser.parse_args()


def create_albumentations_transform() -> A.Compose:
    return A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            # A.Affine(
            #     rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.5
            # ),
            # A.ToGray(p=0.1),
            # A.GaussNoise((0.05, 0.1), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.RandomResizedCrop(
            #     (640, 640), (0.8, 1.0), p=0.4
            # ),  # Keep close to original size
            A.Resize(height=480, width=640),  # Ensure 640x480 for YOLO
            # A.Rotate((-30, 90), p=0.5),
            # A.VerticalFlip(p=0.3),
            # A.HueSaturationValue(
            #     hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
            # ),
            A.Blur(blur_limit=3, p=0.2),
            # A.CoarseDropout(
            #     (8, 8), hole_height_range=(32, 32), hole_width_range=(32, 32), p=0.3
            # ),
        ]
    )


def generate_unique_filename(original_filename: str, counter: int) -> str:
    name, ext = os.path.splitext(original_filename)
    return f"{name}_aug_{counter:03d}_{uuid.uuid4().hex[:8]}.png"


def augment_keurig_image(
    input_dir: Path, output_dir: Path, num_images: int
) -> Dict[str, List[torch.Tensor]]:
    # Define augmentation transforms
    transform = create_albumentations_transform()

    # Get all image files from the input directory
    image_files = get_image_files(input_dir)
    augmented_images = {}

    with Progress() as progress:
        task = progress.add_task(
            "[yellow]Augmenting images...", total=len(image_files) * (num_images + 1)
        )
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            image = read_image(input_path).permute(1, 2, 0).numpy()
            augmented_images[image_file] = [image_file]
            output_path = output_dir / image_file

            progress.update(task, advance=1)
            write_png(torch.from_numpy(image).permute(2, 0, 1), str(output_path))

            for i in range(num_images):
                augmented_image = transform(image=image)["image"]
                unique_filename = generate_unique_filename(image_file, i + 1)
                output_path = output_dir / unique_filename
                # augmented_images[image_file].append((unique_filename, augmented_image))
                write_png(
                    torch.from_numpy(augmented_image).permute(2, 0, 1), str(output_path)
                )
                augmented_images[image_file].append(unique_filename)
                progress.update(task, advance=1)

    console.log(
        f"[bold green]Successfully generated {num_images} augmented images.  Images saved in {output_dir}[/bold green]"
    )
    return augmented_images


def get_image_files(directory: Path) -> List[str]:
    return [
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def split_dataset(
    augmented_images: Dict[str, List[torch.Tensor]],
    output_base_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    console.log("[bold]Splitting datasets...[/bold]")

    # Get all images
    all_images = [
        filename for img_list in augmented_images.values() for filename in img_list
    ]

    # Split the dataset
    train_data, test_val_data = train_test_split(
        all_images, test_size=(1 - train_ratio), random_state=42
    )
    val_data, test_data = train_test_split(
        test_val_data,
        test_size=(test_ratio / (test_ratio + val_ratio)),
        random_state=42,
    )

    # Move files to respective directories
    with Progress() as progress:
        train_task = progress.add_task(
            "[yellow]Saving train data", total=len(train_data)
        )
        val_task = progress.add_task(
            "[yellow]Saving validation data", total=len(val_data)
        )
        test_task = progress.add_task("[yellow]Saving test data", total=len(test_data))

        for data, split, task in zip(
            [train_data, val_data, test_data],
            ["train", "val", "test"],
            [train_task, val_task, test_task],
        ):
            split_dir = output_base_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for filename in data:
                src = output_base_dir / filename
                # output_path = split_dir / filename
                dst = split_dir / filename
                os.rename(src, dst)
                # write_png(img, str(output_path))
                progress.update(task, advance=1)

    console.log(
        f"[bold green]Successfully split datasets. Images in train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}[/bold green]"
    )


def main():
    args = parse_arguments()

    augmented_images = augment_keurig_image(
        args.path, args.output, args.num_augmentations
    )

    # Split the agumented dataset
    split_dataset(
        augmented_images, args.output, args.train_ratio, args.val_ratio, args.test_ratio
    )


if __name__ == "__main__":
    main()
