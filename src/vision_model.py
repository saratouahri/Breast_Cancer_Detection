import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
def get_resnet18(weights=ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model
