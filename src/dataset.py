import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class BreastCancerDataset(Dataset):
    def __init__(self, csv_path, images_dir, tabular_scaler, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.scaler = tabular_scaler
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        # Préparer les features tabulaires
        self.X_tab = pd.DataFrame(self.scaler.transform(
            self.df.drop('target',axis=1)), columns=self.df.columns[:-1]
        )
        self.y = self.df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_id'] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tab = self.X_tab.iloc[idx].values.astype('float32')
        label = self.y[idx]
        return image, tab, label