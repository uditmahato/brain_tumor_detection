import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# Set Streamlit page configuration
st.set_page_config(page_title="MRI Tumor Classification", layout="centered")

# Define the model architecture
class BrainTumorModel(nn.Module):
    def __init__(self):
        super(BrainTumorModel, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.classifier[1].in_features, 4)  # Adjust for 4 output classes
        )

    def forward(self, x):
        return self.base_model(x)

# Load the model
@st.cache_resource
def load_model():
    model = BrainTumorModel()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # EfficientNet normalization
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Class labels
class_labels = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# Streamlit UI
st.title("MRI Tumor Classification")
st.markdown("""
### About the App
This application uses a PyTorch-based EfficientNet model to classify MRI brain scans into one of the following categories:
- **Glioma Tumor**
- **No Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**

Upload an MRI scan image, and the model will analyze and predict the tumor type.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # Convert image to RGB
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing the image..."):
        preprocessed_image = preprocess_image(image)
        with torch.no_grad():
            prediction = model(preprocessed_image)
            predicted_class = torch.argmax(prediction, dim=1).item()

    # Debugging: Display raw predictions
    st.markdown(f"### Raw Model Predictions:\n{prediction.tolist()[0]}")

    # Display the result
    st.success(f"Prediction: **{class_labels[predicted_class]}**")

else:
    st.info("Please upload an MRI scan image to start the classification.")
