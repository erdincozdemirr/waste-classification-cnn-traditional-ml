from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GenericImageDataset(Dataset):
    def __init__(self, data, resize_dim=(224, 224), transform=None):
        """
        Args:
            data (list): (image_path, class_idx) ikililerinden oluşan liste
            resize_dim (tuple): (width, height) şeklinde hedef boyut
            transform (callable, optional): torchvision transformları
        """
        self.data = data
        self.resize_dim = resize_dim
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # AlexNet'in beklentisi
            std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image: {img_path} | Error: {e}")
            raise e

        image = self.transform(image)
        return image, label
