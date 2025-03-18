import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

# -------------------------------
# Updated Dataset Definition for Nested Structure
# -------------------------------
class MapillaryDataset(Dataset):
    """
    Custom dataset that recursively searches a root directory for raw.csv files.
    Each raw.csv is expected to have a header:
      key,lon,lat,ca,captured_at,pano
    For each row, it constructs the image file path by assuming the image is stored in
    the same directory as raw.csv with filename `<key>.jpg`.
    The GPS coordinates are read from the 'lat' and 'lon' fields.
    """
    def __init__(self, photo_dir, meta_dir, transform=None):
        self.transform = transform
        self.data = []  # list of tuples: (image_path, (lat, lon))
        
        # Walk through the root directory recursively.
        for subdir, dirs, files in os.walk(meta_dir):
            if "raw.csv" in files:
                csv_path = os.path.join(subdir, "raw.csv")
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row["key"]
                        try:
                            lon = float(row["lon"])
                            lat = float(row["lat"])
                        except ValueError:
                            continue  # skip if conversion fails
                        # Construct full image path (assumes image filename is key + '.jpg')
                        image_path = os.path.join(subdir.replace('/metadata/', '/') + '/images/', key + ".jpg")

                        if os.path.exists(image_path) and key[:1] != 'a':
                            self.data.append((image_path, (lat, lon)))
                        else:
                            print("Image not found:", image_path)
        random.shuffle(self.data)
        self.data = self.data[:200000]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, coords = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Create tensor for coordinates (order: [lat, lon])
        coords_tensor = torch.tensor(coords, dtype=torch.float)
        return image, coords_tensor

# -------------------------------
# Model definition remains similar
# -------------------------------
class GPSNet(nn.Module):
    """
    Neural network to predict GPS coordinates (latitude, longitude) from an image.
    Uses a pretrained ResNet18 and replaces its final fully connected layer.
    """
    def __init__(self):
        super(GPSNet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 2)  # output: [latitude, longitude]
    
    def forward(self, x):
        return self.base_model(x)

# -------------------------------
# Haversine loss function
# -------------------------------
def haversine_loss(pred, target):
    """
    Computes the mean haversine distance (in kilometers) between predicted
    and target GPS coordinates.
    Assumes that the inputs are in degrees.
    """
    R = 6371.0  # Earth's radius in kilometers
    
    # Convert degrees to radians
    pred_rad = torch.deg2rad(pred)
    target_rad = torch.deg2rad(target)
    
    # True coordinates: lat1, lon1; Predictions: lat2, lon2.
    lat1 = target_rad[:, 0]
    lat2 = pred_rad[:, 0]
    dlat = lat2 - lat1
    dlon = pred_rad[:, 1] - target_rad[:, 1]
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return (R * c).mean()

# -------------------------------
# Training loop with live loss visualization
# -------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_losses = []
    val_losses = []
    
    plt.ion()  # Enable interactive mode for live plot updates
    fig, ax = plt.subplots()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = haversine_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * images.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluate on validation set
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = haversine_loss(outputs, targets)
                epoch_val_loss += loss.item() * images.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f} km, Val Loss: {epoch_val_loss:.4f} km")
        
        # Live update of the loss plot
        ax.clear()
        ax.plot(range(1, epoch+2), train_losses, label="Train Loss")
        ax.plot(range(1, epoch+2), val_losses, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (km)")
        ax.legend()
        plt.draw()
        plt.pause(0.01)
    
    plt.ioff()
    plt.show()

# -------------------------------
# Main function: prepare data, train, and test
# -------------------------------
def main():
    # Root directory that contains the nested structure of CSVs and images.
    # For example, use the train_val directory of your mapillary imagery:
    photo_dir = "/home/derick-yang/Documents/mlsandbox/data/mapillary-imagery/train_val"
    meta_dir = "/home/derick-yang/Documents/mlsandbox/data/mapillary-imagery/metadata/train_val"
    
    # Define image transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create the dataset by scanning the nested directories.
    dataset = MapillaryDataset(photo_dir, meta_dir, transform=transform)
    
    print(f"Found {len(dataset)} images with GPS annotations.")
    
    # Split into train (70%), validation (15%), and test (15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # DataLoaders (adjust num_workers and batch_size as needed)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)
    
    # Set up device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Initialize the model and move it to the device
    model = GPSNet().to(device)
    
    # Train the model
    num_epochs = 25
    train_model(model, train_loader, val_loader, device, num_epochs)
    
    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    # with torch.no_grad():
    #     for images, targets in test_loader:
    #         images = images.to(device)
    #         targets = targets.to(device)
    #         outputs = model(images)
    #         loss = haversine_loss(outputs, targets)
    #         test_loss += loss.item() * images.size(0)
    # test_loss /= len(test_loader.dataset)
    # print(f"Test Loss: {test_loss:.4f} km")
    
    # Optionally, save the trained model
    torch.save(model.state_dict(), "gps_model.pth")
    print("Model saved to gps_model.pth")

if __name__ == "__main__":
    main()
