from collections import Counter
import numpy as np

def print_dataset_distribution(dataset, class_names):
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)

    print("📊 Dataset sınıf dağılımı:")
    for idx, class_name in enumerate(class_names):
        count = label_counts.get(idx, 0)
        print(f"{class_name} ({idx}): {count} örnek")
