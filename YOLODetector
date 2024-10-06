import streamlit as st
from PIL import Image
from ultralytics import YOLO
import re

# Function to load the YOLO model
def load_yolo_model(model_path):
    return YOLO(model_path)

# Function to predict objects in the image
def predict_image(model, image):
    return model(image)

# Sort detections based on x-coordinate
def sort_detections_by_x(detections):
    return sorted(detections, key=lambda x: x[1])  # Sort by the x-coordinate

# Streamlit app
def main():
    st.title("YOLOv11 Image Detector")

    # Regex pattern to filter classes (you can modify this pattern)
    regex_pattern = r'.*rest.*'  # Example: only classes with "rest"

    # Upload image(s)
    uploaded_files = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        model = load_yolo_model("<.pt file here>")  # Load the model once

        for idx, uploaded_file in enumerate(uploaded_files):
            # Each image and its detections will be in a vertical stack
            st.header(f'Image #{idx + 1}:')
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Uploaded Image #{idx + 1}', use_column_width=True)

            # Perform detection
            predictions = predict_image(model, image)

            # Display detection results and allow editing of classes
            st.subheader("Detected Objects:")
            detections = []

            # Collect all detections and their coordinates
            for i in range(len(predictions[0].boxes)):
                cls = int(predictions[0].boxes.cls[i].item())
                class_name = model.names[cls]
                x_coord = predictions[0].boxes.xywhn[i][0].item()
                detections.append((class_name, x_coord))

            # Sort detections from left to right based on x-coordinate
            sorted_detections = sort_detections_by_x(detections)

            # Plot and display the detection image with the original detections
            for r in predictions:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                jpg_file_path = f'results_{idx}.jpg'
                im.save(jpg_file_path)
                st.image(im, caption=f'Detection Result for Image #{idx + 1}', use_column_width=True)

            # Display sorted detections as text
            st.text(" | ".join([f"Class: {det[0]}, X: {det[1]:.2f}" for det in sorted_detections]))

            # Allow the user to edit the predicted class via a dropdown with regex-filtered options
            st.subheader("Edit Predictions:")
            filtered_classes = [name for name in model.names.values() if re.match(regex_pattern, name)]

            for i, (class_name, x_coord) in enumerate(sorted_detections):
                # Create a selectbox for each prediction to allow user editing
                edited_class = st.selectbox(
                    f"Edit Prediction {i + 1} for Image #{idx + 1}",
                    options=filtered_classes if class_name in filtered_classes else model.names.values(),
                    index=filtered_classes.index(class_name) if class_name in filtered_classes else 0
                )
                # Update the detection with the edited class
                sorted_detections[i] = (edited_class, x_coord)

if __name__ == "__main__":
    main()
