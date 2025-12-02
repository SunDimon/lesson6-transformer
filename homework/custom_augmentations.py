import os
import sys
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics'))
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, Solarize, AutoContrast


class RandomGaussianBlur:
    def __call__(self, tensor):
        from torchvision.transforms.functional import to_pil_image, to_tensor
        from PIL import ImageFilter
        img = to_pil_image(tensor)
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        return to_tensor(img)


class RandomPerspective:
    def __call__(self, tensor):
        from torchvision.transforms.functional import to_pil_image, to_tensor
        from torchvision.transforms import RandomPerspective as TVPersp
        img = to_pil_image(tensor)
        persp = TVPersp(distortion_scale=0.3, p=1.0)
        img = persp(img)
        return to_tensor(img)


class RandomBrightnessContrast:
    def __call__(self, tensor):
        from torchvision.transforms.functional import to_pil_image, to_tensor
        from PIL import ImageEnhance
        img = to_pil_image(tensor)
        img = ImageEnhance.Brightness(img).enhance(1.5)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        return to_tensor(img)


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)


def visualize_custom_augmentations():
    current_dir = Path(__file__).parent
    root = current_dir / ".." / "data" / "train"
    root = str(root.resolve())
    dataset = CustomImageDataset(root, transform=transforms.ToTensor(), target_size=(224, 224))
    class_names = dataset.get_class_names()
    img_tensor, label = dataset[170]
    class_name = class_names[label]

    orig = img_tensor.clone()

    extra_augs = [
        ("Гауссов шум", AddGaussianNoise(mean=0.0, std=0.1)),
        ("Соляризация", Solarize(threshold=100)),
        ("Автоконтраст", AutoContrast(p=1.0))
    ]

    custom_augs = [
        ("Размытие", RandomGaussianBlur()),
        ("Перспектива", RandomPerspective()),
        ("Яркость+Контраст", RandomBrightnessContrast())
    ]

    extra_results = [(name, aug(orig.clone())) for name, aug in extra_augs]
    custom_results = [(name, aug(orig.clone())) for name, aug in custom_augs]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Сравнение аугментаций — класс: {class_name}", fontsize=16)

    axes[0, 0].imshow(tensor_to_numpy(orig))
    axes[0, 0].set_title("Оригинал", fontsize=12)
    axes[0, 0].axis('off')

    for j, (name, img) in enumerate(extra_results, 1):
        axes[0, j].imshow(tensor_to_numpy(img))
        axes[0, j].set_title(name, fontsize=12)
        axes[0, j].axis('off')

    axes[1, 0].imshow(tensor_to_numpy(orig))
    axes[1, 0].set_title("Оригинал", fontsize=12)
    axes[1, 0].axis('off')

    for j, (name, img) in enumerate(custom_results, 1):
        axes[1, j].imshow(tensor_to_numpy(img))
        axes[1, j].set_title(name, fontsize=12)
        axes[1, j].axis('off')

    plt.tight_layout()
    plt.show()


# visualize_custom_augmentations()