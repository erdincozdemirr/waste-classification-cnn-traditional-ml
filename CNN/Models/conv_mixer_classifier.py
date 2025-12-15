import torch
import torch.nn as nn
import timm

class ConvMixerClassifier(nn.Module):
    def __init__(self, num_classes=6, model_name='convmixer_768_32', pretrained=True):
        super(ConvMixerClassifier, self).__init__()

        # Modeli yükle
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Son katmanı num_classes'a göre güncelle
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"{model_name} modelinde 'classifier' veya 'head' bulunamadı.")

    def forward(self, x):
        return self.backbone(x)
