import torch.nn as nn
import torchvision.models as models

class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(AlexNetClassifier, self).__init__()
        self.backbone = models.alexnet(pretrained=pretrained)
        in_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
