
# multimodal_model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
class MultimodalNet(nn.Module):
    def __init__(self, tabular_input_dim=30, tabular_hidden_dims=[64, 32]):
        super(MultimodalNet, self).__init__()

        # Vision : ResNet18 sans la dernière couche
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # sortie = 512

        # Tabulaire : MLP
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(tabular_hidden_dims[0], tabular_hidden_dims[1]),
            nn.ReLU()
        )

        # Fusion + prédiction
        self.classifier = nn.Sequential(
            nn.Linear(512 + tabular_hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, tabular):
        image_feat = self.cnn(image)                 # (B, 512)
        tabular_feat = self.tabular_net(tabular)     # (B, 32)
        combined = torch.cat((image_feat, tabular_feat), dim=1)
        output = self.classifier(combined)
        return output
