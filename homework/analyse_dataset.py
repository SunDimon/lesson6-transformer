import os
import sys
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics'))
from datasets import CustomImageDataset


def analyse_dataset():
    current_dir = Path(__file__).parent
    root = current_dir / ".." / "data" / "train"
    root = str(root.resolve())
    
    dataset = CustomImageDataset(root, transform=None, target_size=None)
    class_names = dataset.get_class_names()
    
    print(f"Всего изображений: {len(dataset)}\n")

    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    class_counts_named = {class_names[k]: v for k, v in sorted(class_counts.items())}
    for cls, count in class_counts_named.items():
        print(f"Класс '{cls}': {count} изображений")

    all_widths = []
    all_heights = []
    class_sizes = defaultdict(lambda: {"widths": [], "heights": []})

    for i in range(len(dataset)):
        img, label = dataset[i]
        cls_name = class_names[label]

        w, h = img.size

        all_widths.append(w)
        all_heights.append(h)
        class_sizes[cls_name]["widths"].append(w)
        class_sizes[cls_name]["heights"].append(h)

    min_w, max_w = min(all_widths), max(all_widths)
    min_h, max_h = min(all_heights), max(all_heights)
    avg_w = sum(all_widths) / len(all_widths)
    avg_h = sum(all_heights) / len(all_heights)
    print()
    print(f"Ширина:  мин={min_w}, макс={max_w}, сред={avg_w:.1f}")
    print(f"Высота:  мин={min_h}, макс={max_h}, сред={avg_h:.1f}")

    n_classes = len(class_names)
    cols = 2
    rows = (n_classes + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, cls_name in enumerate(class_names):
        ax = axes[idx]
        widths = class_sizes[cls_name]["widths"]
        heights = class_sizes[cls_name]["heights"]

        ax.hist(widths, bins=30, alpha=0.7, label='Ширина', color='green')
        ax.hist(heights, bins=30, alpha=0.7, label='Высота', color='orange')
        ax.set_title(f'Размеры изображений — класс "{cls_name}"')
        ax.set_xlabel('Пиксели')
        ax.set_ylabel('Частота')
        ax.legend()
        ax.grid(alpha=0.3)

    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# analyse_dataset()