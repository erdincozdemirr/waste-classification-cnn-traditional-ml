import os
from PIL import Image

def find_min_image_size(root_dir):
    """
    Belirtilen dizindeki tüm alt klasörlerdeki görselleri tarar
    ve en küçük (width, height) boyutunu döner.

    Args:
        root_dir (str): Ana veri klasör yolu (örneğin: 'Datasets')

    Returns:
        tuple: (min_width, min_height)
    """
    min_width, min_height = float('inf'), float('inf')

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    min_width = min(min_width, width)
                    min_height = min(min_height, height)
            except Exception as e:
                print(f"Hata: {img_path} dosyasında {e}")

    return (min_width, min_height)
