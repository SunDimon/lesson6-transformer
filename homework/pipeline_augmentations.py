import os
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance


class AugmentationPipeline:
    def __init__(self):
        self._augmentations = {}

    def add_augmentation(self, name, aug):
        self._augmentations[name] = aug

    def remove_augmentation(self, name):
        if name in self._augmentations:
            del self._augmentations[name]
        else:
            raise KeyError(f"Augmentation '{name}' not found.")

    def apply(self, image):
        img = image.copy()
        for aug in self._augmentations.values():
            img = aug(img)
        return img

    def get_augmentations(self):
        return self._augmentations.copy()


def random_horizontal_flip(p=0.5):
    def aug(img):
        return ImageOps.mirror(img) if random.random() < p else img
    return aug


def random_brightness(factor_range=(0.8, 1.2)):
    def aug(img):
        factor = random.uniform(*factor_range)
        return ImageEnhance.Brightness(img).enhance(factor)
    return aug


def random_contrast(factor_range=(0.8, 1.2)):
    def aug(img):
        factor = random.uniform(*factor_range)
        return ImageEnhance.Contrast(img).enhance(factor)
    return aug


def random_saturation(factor_range=(0.8, 1.2)):
    def aug(img):
        factor = random.uniform(*factor_range)
        return ImageEnhance.Color(img).enhance(factor)
    return aug


def random_rotation(degrees=10):
    def aug(img):
        angle = random.uniform(-degrees, degrees)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    return aug


def get_light_pipeline():
    p = AugmentationPipeline()
    p.add_augmentation("hflip", random_horizontal_flip(0.5))
    p.add_augmentation("brightness", random_brightness((0.9, 1.1)))
    return p


def get_medium_pipeline():
    p = AugmentationPipeline()
    p.add_augmentation("hflip", random_horizontal_flip(0.5))
    p.add_augmentation("brightness", random_brightness((0.8, 1.2)))
    p.add_augmentation("contrast", random_contrast((0.8, 1.2)))
    p.add_augmentation("rotation", random_rotation(5))
    return p


def get_heavy_pipeline():
    p = AugmentationPipeline()
    p.add_augmentation("hflip", random_horizontal_flip(0.5))
    p.add_augmentation("brightness", random_brightness((0.7, 1.3)))
    p.add_augmentation("contrast", random_contrast((0.7, 1.3)))
    p.add_augmentation("saturation", random_saturation((0.7, 1.3)))
    p.add_augmentation("rotation", random_rotation(15))
    return p


def apply_and_save_config(train_root, output_root, pipeline, config_name):
    train_path = Path(train_root)
    output_path = Path(output_root) / config_name
    output_path.mkdir(parents=True, exist_ok=True)

    image_count = 0
    for class_dir in train_path.iterdir():
        if not class_dir.is_dir():
            continue
        (output_path / class_dir.name).mkdir(exist_ok=True)
        for img_file in class_dir.iterdir():
            img = Image.open(img_file).convert("RGB")
            aug_img = pipeline.apply(img)
            save_path = output_path / class_dir.name / img_file.name
            aug_img.save(save_path, quality=95)
            image_count += 1


def apply_augmentations():
    current_dir = Path(__file__).parent
    train_root = current_dir / ".." / "data" / "train"
    train_root = str(train_root.resolve())
    output_root = "augmented"

    configs = {
        "light": get_light_pipeline(),
        "medium": get_medium_pipeline(),
        "heavy": get_heavy_pipeline(),
    }

    for name, pipeline in configs.items():
        apply_and_save_config(train_root, output_root, pipeline, name)


# apply_augmentations()