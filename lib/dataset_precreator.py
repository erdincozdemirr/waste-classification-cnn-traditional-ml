import os
import random


def get_images_by_class(root_dir, max_per_class=837):
    """
    Her sınıftan rastgele 'max_per_class' kadar görsel alır.

    Returns:
        dict: {class_name: [image_path1, image_path2, ...]}
        list: class_names
    """
    class_image_dict = {}
    class_names = sorted([
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name)) and not name.startswith('.')
    ])

    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        image_list = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg')) and not img.startswith('.')
        ]

        if len(image_list) >= max_per_class:
            selected = random.sample(image_list, max_per_class)
        else:
            print(f"UYARI: {class_name} sınıfında {len(image_list)} görsel var.")
            selected = image_list

        class_image_dict[class_name] = selected

    return class_image_dict, class_names
    """
    Her sınıftan 'max_per_class' kadar görsel alır ve class index ile eşleştirir.

    Returns:
        dict: {class_name: [image_path1, image_path2, ...]}
    """
    class_image_dict = {}
    class_names = sorted(os.listdir(root_dir))  # alfabetik sıraya göre class index verelim

    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_list = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(image_list) >= max_per_class:
            selected = random.sample(image_list, max_per_class)
        else:
            print(f"UYARI: {class_name} sınıfında {len(image_list)} görsel var, {max_per_class} yerine onları alıyoruz.")
            selected = image_list

        class_image_dict[class_name] = selected

    return class_image_dict, class_names  # class_names sırası index için lazım olacak

from sklearn.model_selection import train_test_split

def split_data(image_dict, class_names, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Her sınıftan image-path ve class-index ikilileri oluşturur ve train/val/test olarak böler.

    Returns:
        train_data, val_data, test_data : list of (path, class_idx)
    """
    train_data, val_data, test_data = [], [], []

    for idx, class_name in enumerate(class_names):
        images = image_dict[class_name]
        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - split_ratio[0]), random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)

        train_data += [(img_path, idx) for img_path in train_imgs]
        val_data   += [(img_path, idx) for img_path in val_imgs]
        test_data  += [(img_path, idx) for img_path in test_imgs]

    return train_data, val_data, test_data

def flatten_image_dict(image_dict, class_names):
    image_paths = []
    labels = []

    for idx, class_name in enumerate(class_names):
        for path in image_dict[class_name]:
            image_paths.append(path)
            labels.append(idx)

    return image_paths, labels

