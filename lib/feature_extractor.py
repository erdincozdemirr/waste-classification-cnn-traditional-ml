import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.models as models
from torchvision import transforms


class FeatureDataset(Dataset):
    def __init__(self, image_paths, labels, extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Mobilenet ile feature çıkar (numpy)
        feature = self.extractor.extract(image_path)

        # Tensor'a çevir
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor


class ImageFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.mobilenet_v2(pretrained=True).features.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)
            return torch.flatten(features, 1).cpu().numpy().squeeze()
