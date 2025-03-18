import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import folium
import numpy as np
import base64

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
# Main prediction and multi-marker visualization function
# -------------------------------
def main(list_filename):
    # Read the newline-separated list of JPEG paths
    try:
        with open(list_filename, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip() and 'images/a' in line]
    except Exception as e:
        print(f"Error reading file {list_filename}: {e}")
        sys.exit(1)

    if not image_paths:
        print("No image paths found in the list.")
        sys.exit(1)

    # Define transforms (should match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Set up device and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPSNet().to(device)
    try:
        model.load_state_dict(torch.load("gps_model.pth", map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    
    # Store predicted coordinates and associated popup HTML for markers
    markers = []
    
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            continue
        
        input_tensor = transform(image).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            prediction = model(input_tensor.to(device))
        prediction = prediction.cpu().numpy()[0]
        predicted_lat, predicted_lon = prediction[0], prediction[1]
        
        # Read image file in binary mode and encode to base64 for embedding in the popup
        try:
            with open(path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
            html_popup = f"""
                <div style="width:220px">
                    <p>{path}<br>({predicted_lat:.6f}, {predicted_lon:.6f})</p>
                    <img src="data:image/jpeg;base64,{encoded}" style="width:200px;">
                </div>
            """
        except Exception as e:
            print(f"Error encoding image {path}: {e}")
            html_popup = f"{path}\n({predicted_lat:.6f}, {predicted_lon:.6f})"
        
        markers.append((predicted_lat, predicted_lon, html_popup))
        print(f"{path}: Predicted (lat, lon) = ({predicted_lat:.6f}, {predicted_lon:.6f})")
    
    if not markers:
        print("No valid predictions could be made.")
        sys.exit(1)
    
    # Compute center of the map as the average of all predicted coordinates
    lats = [m[0] for m in markers]
    lons = [m[1] for m in markers]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create a Folium map centered at the computed average location
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    for lat, lon, popup_html in markers:
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="red")
        ).add_to(m)
    
    # Save the map to an HTML file
    map_filename = f"predicted_locations{list_filename.replace('.', '')[:5]}.html"
    m.save(map_filename)
    print(f"Map saved to {map_filename}. Open this file in your browser to view the predictions.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_multiple_predictions.py <path_to_list_file>")
        sys.exit(1)
    list_filename = sys.argv[1]
    main(list_filename)
