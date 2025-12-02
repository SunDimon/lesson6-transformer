import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics'))
from datasets import CustomImageDataset


def visualize_augmentations_pipeline():
    current_dir = Path(__file__).parent
    root = current_dir / ".." / "data" / "train"
    root = root.resolve()
    dataset = CustomImageDataset(str(root), transform=None, target_size=(224, 224))
    class_names = dataset.get_class_names()

    selected_images = []
    selected_labels = []
    selected_classes = set()
    
    for i in range(len(dataset)):
        if len(selected_images) >= 5:
            break
        img, label = dataset[i]
        class_name = class_names[label]
        if class_name not in selected_classes:
            selected_images.append(img)
            selected_labels.append(label)
            selected_classes.add(class_name)
    
    standard_augmentations = [
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomCrop", transforms.RandomCrop(180, padding=20)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ("RandomRotation", transforms.RandomRotation(degrees=30)),
        ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
    ]
    
    def plot_augmentations(original_imgs, augmented_imgs_list, titles, main_title):
        n_originals = len(original_imgs)
        n_augmentations = len(augmented_imgs_list)
        
        fig, axes = plt.subplots(n_augmentations + 1, n_originals, 
                                figsize=(n_originals * 2, (n_augmentations + 1) * 2))
        
        if n_originals == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img in enumerate(original_imgs):
            if isinstance(img, Image.Image):
                img_np = np.array(img)
            else:
                img_np = img.numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"\n\n\n{class_names[selected_labels[i]]}", fontsize=10)
            axes[0, i].axis('off')
        
        for row, (aug_imgs, title) in enumerate(zip(augmented_imgs_list, titles), 1):
            for col, img in enumerate(aug_imgs):
                if isinstance(img, Image.Image):
                    img_np = np.array(img)
                else:
                    img_np = img.numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np, 0, 1)
                
                axes[row, col].imshow(img_np)
                if col == 0:
                    axes[row, col].set_ylabel(title, rotation=90, fontsize=10)
                axes[row, col].axis('off')
        
        plt.suptitle(main_title, fontsize=14, y=0.95)
        plt.tight_layout()
        plt.show()
    
    fig, axes = plt.subplots(1, len(selected_images), figsize=(15, 3))
    
    if len(selected_images) == 1:
        axes = [axes]
    
    for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
        img_np = np.array(img)
        axes[i].imshow(img_np)
        axes[i].set_title(f"{class_names[label]}", fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle("Оригинальные изображения из разных классов", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    all_augmented_results = []
    augmentation_names = []
    
    for aug_name, augmentation in standard_augmentations:
        augmented_imgs = []
        for img in selected_images:
            if aug_name in ["RandomHorizontalFlip", "RandomCrop", "RandomRotation", "RandomGrayscale"]:
                aug_img = augmentation(img)
                augmented_imgs.append(aug_img)
            else:
                aug_img = augmentation(img)
                augmented_imgs.append(aug_img)
        
        all_augmented_results.append(augmented_imgs)
        augmentation_names.append(aug_name)
    
    plot_augmentations(selected_images, all_augmented_results, augmentation_names,
                      "Результат применения каждой аугментации отдельно")

    full_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Pad(padding=20, fill=128),
        transforms.RandomCrop(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomRotation(degrees=(30, 30)),
        transforms.RandomGrayscale(p=1.0),
        transforms.Resize((224, 224))
    ])

    fully_augmented = []
    to_tensor = transforms.ToTensor()

    for img in selected_images:
        aug_img = full_pipeline(img)
        aug_tensor = to_tensor(aug_img)
        fully_augmented.append(aug_tensor)

    fig, axes = plt.subplots(2, len(selected_images), figsize=(len(selected_images) * 2.5, 6))

    if len(selected_images) == 1:
        axes = axes.reshape(2, 1)

    for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
        img_tensor = to_tensor(img)
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Оригинал\n{class_names[label]}", fontsize=10)
        axes[0, i].axis('off')

    for i, img in enumerate(fully_augmented):
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axes[1, i].imshow(img_np)
        if i == 0:
            axes[1, i].set_ylabel("Все 5 аугментаций", rotation=90, fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle("Оригинал и результат после последовательного применения всех 5 аугментаций", fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()


# visualize_augmentations_pipeline()