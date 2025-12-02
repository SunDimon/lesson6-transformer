import os
import random
import time
import tracemalloc
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt


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


def random_rotation(degrees=10):
    def aug(img):
        angle = random.uniform(-degrees, degrees)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    return aug


def build_pipeline():
    pipeline = []
    pipeline.append(random_horizontal_flip(0.5))
    pipeline.append(random_brightness((0.8, 1.2)))
    pipeline.append(random_contrast((0.8, 1.2)))
    pipeline.append(random_rotation(10))
    return pipeline


def load_sample_images(root: str, n: int = 100):
    root_path = Path(root)
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []

    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(f)
                if len(image_paths) >= n:
                    return image_paths
    return image_paths[:n]


def benchmark_size(size, image_paths, pipeline):
    target_size = (size, size)
    total_time = 0.0
    peak_memory = 0

    random.seed(42)

    resized_images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        resized = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(resized)

    tracemalloc.start()
    start_time = time.perf_counter()

    for img in resized_images:
        out = img
        for aug in pipeline:
            out = aug(out)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end_time - start_time
    peak_memory = peak / (1024 * 1024)

    return total_time, peak_memory


def experiment_with_sizes():
    current_dir = Path(__file__).parent
    train_root = current_dir / ".." / "data" / "train"
    train_root = train_root.resolve()

    image_paths = load_sample_images(str(train_root), n=100)

    sizes = [64, 128, 224, 512]
    pipeline = build_pipeline()

    times = []
    memories = []

    for size in sizes:
        print(f"  Тестируем размер: {size}x{size}")
        t, mem = benchmark_size(size, image_paths, pipeline)
        times.append(t)
        memories.append(mem)
        print(f"    Время: {t:.3f} сек, Память: {mem:.2f} MB")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Размер изображения (px)')
    ax1.set_ylabel('Время обработки 100 изображений (сек)', color=color)
    ax1.plot(sizes, times, 'o-', color=color, label='Время')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Пиковое потребление памяти (MB)', color=color)
    ax2.plot(sizes, memories, 's--', color=color, label='Память')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Зависимость времени и памяти от размера изображения\n(100 изображений, medium аугментации)')
    fig.tight_layout()
    plt.show()


# experiment_with_sizes()