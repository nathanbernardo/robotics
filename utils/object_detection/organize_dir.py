import os
import shutil
from pathlib import Path
import yaml
import argparse
import glob
from rich.progress import track
from rich.console import Console

console = Console()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program to organizing dataset directory for Ultralytics")
    parser.add_argument("--path", type=Path, default=Path('.'), help="Path to the target directory (default: current directory)")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training dataset")
    parser.add_argument("--val", type=Path, required=True, help="Path to the validation dataset")
    parser.add_argument("--train_images", type=Path, required=True, help="Path to images for training dataset")
    parser.add_argument("--validation_images", type=Path, required=True, help="Path to images for validation dataset")

    return parser.parse_args()


# Create directories for storing image dataset
def create_directories(base_path):
    console.log("Creating directories...")
    for folder in ['train', 'val']:
        for subdir in ['images', 'labels']:
            dir_path = base_path / folder / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            console.log(f"Created directory: [bold green]{dir_path}")
            # (base_path / folder / subdir).mkdir(parents=True, exist_ok=True)

def merge_txt_files(train_file, validation_file, output_file):
    console.log(f"Merging text files into [bold]{output_file}[/bold]...")
    with open(output_file, 'w') as outfile:
        for fname in [train_file, validation_file]:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())

def copy_labels(src_dir, dst_dir):
    console.log(f"Copying labels from [bold]{src_dir}[/bold] to [bold]{dst_dir}[/bold]...")
    for file in os.listdir(src_dir):
        if file.endswith(".txt"):
            shutil.copy(os.path.join(src_dir, file), dst_dir)
    console.log("[bold green]Labels copied successfully![/bold]")

def copy_images(src_path, dst_dir):
    console.log(f"Copying images from [bold]{src_path}[/bold] to [bold]{dst_dir}[/bold]...")
    for img in glob.glob(os.path.join(src_path, "*.png")):
        shutil.copy(img, dst_dir)
    console.log("[bold green]Images copied successfully![/bold]")

def update_yaml(src_yaml, dst_yaml):
    console.log(f"Updating YAML file: [bold]{dst_yaml}[/bold]...")
    with open(src_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Update the paths
    yaml_data['path'] = '.'
    yaml_data['train'] = 'train/images'
    yaml_data['val'] = 'val/images'
    yaml_data['Train'] = 'Data.txt'
    
    with open(dst_yaml, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

    console.log("[bold green]YAML file successfully updated")

def removed_unlabeled_images(src_dir, label_path):
    console.log(f"Removing unlabeled images from [bold]{src_dir}[/bold]...")
    labeled_images = set(f"{os.path.splitext(label)[0]}.png" for label in os.listdir(label_path) if label.endswith('.txt'))

    # Remove unlabeled images
    for img in track(os.listdir(src_dir), description="Processing images..."):
        if img not in labeled_images:
            os.remove(f"{src_dir}/{img}")
    
    console.log("[bold green]Unlabeled images removed successfully![/bold]")


def main():
    args = parse_arguments()

    base_path = args.path.absolute()
    train_path = args.train.absolute()
    train_images_path = args.train_images.absolute()
    val_path = args.val.absolute()
    val_images_path = args.validation_images.absolute()

    # Step 1: Create necessary directories for storing images and labels
    create_directories(base_path)

    # Step 2: Merge training and validation text files
    merge_txt_files(f"{train_path}/Train.txt", f"{val_path}/Validation.txt", f"{base_path}/Data.txt")

    # Step 3: Copy label files to new directories
    copy_labels(f"{train_path}/labels/Train", f"{base_path}/train/labels")
    copy_labels(f"{val_path}/labels/Validation", f"{base_path}/val/labels")

    # Step 4: Copy image files to new directories
    copy_images(train_images_path, f"{base_path}/train/images")
    copy_images(val_images_path, f"{base_path}/val/images")

    # Step 5: Removed unlabeled images
    removed_unlabeled_images(f"{base_path}/train/images", f"{train_path}/labels/Train")
    removed_unlabeled_images(f"{base_path}/val/images", f"{val_path}/labels/Validation")

    # Step 6: Update YAML configuration file
    update_yaml(f"{train_path}/data.yaml", f"{base_path}/data.yaml")

if __name__ == "__main__":
    main()
