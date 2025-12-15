import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetOverrided(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(AlexNetOverrided, self).__init__()

        # Pretrained modeli al
        base_model = models.alexnet(pretrained=pretrained)

        # Sadece ilk 9 katmanı al (Conv+ReLU+MaxPool vs dahil)
        self.features = nn.Sequential(*list(base_model.features.children())[:9])

        # Flatten sonrası gelen boyutu bilmiyorsan dummy inputla öğren
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
