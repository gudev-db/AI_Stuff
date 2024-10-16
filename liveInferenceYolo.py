import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Load the YOLO model (you can use any pre-trained or custom model here)
model = YOLO('/home/henrique/Desktop/OMR./OMR/recortador/betterCap.pt')

# Streamlit UI setup
st.title("Real-Time Object Detection with YOLO")
st.subheader("Upload a video, image, use webcam, or connect to a remote camera for live inference.")

# Option to select the source: webcam, video, image, or remote camera
source_type = st.selectbox('Select Input Source', ('Webcam', 'Upload Video', 'Upload Image', 'Remote Camera'))

# Variables for video or image source
video_source = None
image_source = None

if source_type == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_source = temp_file.name

elif source_type == 'Upload Image':
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_image.read())
        image_source = temp_file.name  # Use the name of the temporary file

elif source_type == 'Remote Camera':
    remote_url = st.text_input('Enter the remote camera URL (RTSP/HTTP):')
    if remote_url:
        video_source = remote_url

else:
    video_source = 0  # Webcam as source

# Slider for confidence and NMS thresholds
confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
nms_threshold = st.slider('NMS Threshold', 0.0, 1.0, 0.45)

# Initialize counters
class_counter = Counter()

# Placeholders for displaying counts and detected objects
detected_count_placeholder = st.empty()
detected_objects_placeholder = st.empty()
class_count_placeholder = st.sidebar.empty()  # Placeholder for class counts

# Start detection button
if st.button('Start Detection'):

    # If the source is a video, webcam, or remote camera
    if source_type in ['Webcam', 'Upload Video', 'Remote Camera']:
        if video_source is not None:
            cap = cv2.VideoCapture(video_source)

            stframe = st.empty()  # Placeholder to display video frames
            pie_chart_placeholder = st.sidebar.empty()  # Placeholder for pie chart

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Can't read video source.")
                    break

                # YOLOv8 inference on video frame
                results = model(frame, conf=confidence_threshold, iou=nms_threshold)

                detected_classes = []

                # Loop over detected objects
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        detected_classes.append(class_name)

                # Update cumulative class counter
                class_counter.update(detected_classes)

                # Update and display the current count of detected objects in this frame
                current_frame_count = len(detected_classes)
                detected_count_placeholder.write(f"Detected Objects in Frame: {current_frame_count}")

                # Count occurrences of each detected class
                class_count = Counter(detected_classes)

                # Prepare a string to display the counts for each class
                class_count_display = ', '.join([f"{cls}: {count}" for cls, count in class_count.items()])

                # Update and display the count of detected classes in this frame
                detected_objects_placeholder.write(f"Detected Objects: {class_count_display}")

                # Draw boxes and labels on the frame
                annotated_frame = results[0].plot()

                # Display the video frame with detections
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                # Create and display the pie chart with cumulative class distribution
                if class_counter:
                    labels, values = zip(*class_counter.items())
                    fig, ax = plt.subplots()
                    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.axis('equal')
                    pie_chart_placeholder.pyplot(fig)

            cap.release()

    # If the source is an image
    elif source_type == 'Upload Image' and image_source is not None:
        # Read the image using OpenCV
        image = cv2.imread(image_source)  # Now correctly uses the image source path

        # YOLOv8 inference on image
        results = model(image, conf=confidence_threshold, iou=nms_threshold)

        detected_classes = []

        # Loop over detected objects in the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)

        # Update cumulative class counter (optional for image)
        class_counter.update(detected_classes)

        # Update and display the count of detected objects in the image
        current_image_count = len(detected_classes)
        detected_count_placeholder.write(f"Detected Objects in Image: {current_image_count}")

        # Count occurrences of each detected class
        class_count = Counter(detected_classes)

        # Prepare a string to display the counts for each class
        class_count_display = ', '.join([f"{cls}: {count}" for cls, count in class_count.items()])

        # Update and display the count of detected classes in the image
        detected_objects_placeholder.write(f"Detected Objects: {class_count_display}")

        # Draw boxes and labels on the image
        annotated_image = results[0].plot()

        # Display the image with detections
        st.image(annotated_image, channels="BGR", use_column_width=True)

        # Create and display the pie chart with class distribution for the image
        if class_counter:
            labels, values = zip(*class_counter.items())
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')
            st.sidebar.pyplot(fig)
