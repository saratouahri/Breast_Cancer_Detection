# multimodal_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()

        self.image_names = self.data['image_name']  # colonne avec noms d'images
        self.tabular_data = self.data.drop(columns=['image_name', 'target']).values.astype('float32')
        self.labels = self.data['target'].values.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Charger image
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tabulaire
        tabular = torch.tensor(self.tabular_data[idx])

        # Label
        label = torch.tensor([self.labels[idx]])

        return image, tabular, label