import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.dataset import BreastCancerDataset
from src.multimodal_model import MultiModalNet
from src.preprocess import load_data, preprocess_data
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

def train_multimodal(csv_path, images_dir, epochs=10, batch_size=16):
    # 1) Prétraitement tabulaire
    df = load_data(csv_path)
    _, _, _, _, scaler = preprocess_data(df)
    # 2) Dataset & DataLoader
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = BreastCancerDataset(csv_path, images_dir+'/train', scaler, transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # 3) Modèle, loss, opti
    model = MultiModalNet(tab_features=df.shape[1]-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # 4) Training loop
    model.train()
    for epoch in range(epochs):
        for imgs, tabs, labels in train_dl:
            imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(imgs, tabs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    # 5) Sauvegarder le modèle final
    torch.save(model.state_dict(), 'models/multimodal_model.pt')

if __name__ == '__main__':
    train_multimodal('data/breast_cancer.csv', 'data/images')