import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import folium

# -------------------------------
# Define the model architecture (same as during training)
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
# Main prediction and visualization function
# -------------------------------
def main(image_filename):
    # Define transforms (should match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    try:
        image = Image.open(image_filename).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_filename}: {e}")
        sys.exit(1)
    
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
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
    prediction = prediction.cpu().numpy()[0]
    predicted_lat, predicted_lon = prediction[0], prediction[1]
    
    print(f"Predicted Location: Latitude {predicted_lat:.6f}, Longitude {predicted_lon:.6f}")
    
    # Create an interactive map using Folium
    m = folium.Map(location=[predicted_lat, predicted_lon], zoom_start=12)
    folium.Marker(
        location=[predicted_lat, predicted_lon],
        popup="Predicted Location",
        icon=folium.Icon(color="red")
    ).add_to(m)
    
    # Save the map to an HTML file
    map_filename = "predicted_location.html"
    m.save(map_filename)
    print(f"Map saved to {map_filename}. Open this file in your browser to view the prediction.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_prediction.py <image_filename>")
        sys.exit(1)
    image_filename = sys.argv[1]
    main(image_filename)
