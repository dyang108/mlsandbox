import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import textwrap

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)


# Initialize webcam
cap = cv2.VideoCapture(0)

# Text wrapping parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6  # Smaller font
font_thickness = 1
text_color = (0, 0, 0)
line_spacing = 20  # Spacing between lines
max_width = 500  # Maximum width in pixels for wrapping text

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare image for model
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Wrap text to fit within a specified width
    char_per_line = 40  # Adjust based on your frame size
    wrapped_text = textwrap.wrap(caption, width=char_per_line)

    # Display wrapped text on frame
    x, y = 20, 40  # Start position
    for line in wrapped_text:
        cv2.putText(frame, line, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        y += line_spacing  # Move to next line

    # Show the frame
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
