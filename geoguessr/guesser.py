import sys
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import folium
import requests
from io import BytesIO
import webbrowser

# -------------------------------
# Define the model architecture (must match training setup)
# -------------------------------
class GPSNet(nn.Module):
    def __init__(self):
        super(GPSNet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 2)  # outputs [latitude, longitude]
    
    def forward(self, x):
        return self.base_model(x)

# -------------------------------
# Function to load an image from a local path or URL
# -------------------------------
def load_image(image_source):
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # Fetch image from the web
        response = requests.get(image_source)
        if response.status_code != 200:
            print(f"Error downloading image: {response.status_code}")
            sys.exit(1)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Load local image
        if not os.path.exists(image_source):
            print(f"File not found: {image_source}")
            sys.exit(1)
        image = Image.open(image_source).convert("RGB")
    
    return image

# -------------------------------
# Main function to predict location and show map
# -------------------------------
def predict_and_show(image_source):
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = load_image(image_source)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Set up device and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPSNet().to(device)
    try:
        model.load_state_dict(torch.load("gps_model.pth", map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model.eval()

    # Run inference
    with torch.no_grad():
        prediction = model(input_tensor.to(device))
    predicted_lat, predicted_lon = prediction.cpu().numpy()[0]
    
    print(f"Predicted Location: Latitude {predicted_lat:.6f}, Longitude {predicted_lon:.6f}")
    
    # Create interactive map with a marker
    m = folium.Map(location=[predicted_lat, predicted_lon], zoom_start=12)
    folium.Marker(
        location=[predicted_lat, predicted_lon],
        popup=f"Predicted Location\n({predicted_lat:.6f}, {predicted_lon:.6f})",
        icon=folium.Icon(color="red")
    ).add_to(m)
    
    # Save the map and open in browser
    map_filename = "predicted_location.html"
    m.save(map_filename)
    print(f"Map saved to {map_filename}. Opening in browser...")
    webbrowser.open(f"file://{os.path.abspath(map_filename)}")

# -------------------------------
# Command-line interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict location from an image using a trained model and open a map in the browser.")
    parser.add_argument("image_source", help="URL or local path to an image")
    args = parser.parse_args()
    
    predict_and_show(args.image_source)
