import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import numpy as np

# Load the YOLO model
model = YOLO('/home/henrique/Desktop/OMR./OMR/recortador/segment_delone.pt')

# Streamlit UI setup
st.title("Power Line Angle Detector")
source_type = st.selectbox('Select Input Source', ('Webcam', 'Upload Video', 'Upload Image', 'Remote Camera'))

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
        image_source = temp_file.name

elif source_type == 'Remote Camera':
    remote_url = st.text_input('Enter the remote camera URL (RTSP/HTTP):')
    if remote_url:
        video_source = remote_url

else:
    video_source = 0  # Webcam as source

confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
nms_threshold = st.slider('NMS Threshold', 0.0, 1.0, 0.45)

detected_objects_placeholder = st.empty()
detected_objects_info = []  # List to accumulate object detection results

if st.button('Start Detection'):
    if video_source is not None:
        cap = cv2.VideoCapture(video_source)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read video source.")
                break

            results = model(frame, conf=confidence_threshold, iou=nms_threshold)

            for result in results:
                masks = result.masks  # Get segmentation masks
                if masks is not None:
                    for i in range(len(masks)):
                        class_id = int(result.boxes[i].cls[0])
                        class_name = model.names[class_id]

                        # Get the mask points directly
                        mask_points = masks.xy[i]  # NumPy array
                        if mask_points.size > 0:
                            # Draw contours based on the mask points
                            contour = mask_points.astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, contour, isClosed=True, color=(255, 0, 0), thickness=2)

                            # Calculate the coordinates for regression
                            x_coords = mask_points[:, 0]
                            y_coords = mask_points[:, 1]

                            # Fit a polynomial (degree 1 for a linear fit)
                            coeffs = np.polyfit(x_coords, y_coords, 1)  # Degree 1 for linear
                            slope = coeffs[0]  # Coefficient for x (slope)

                            # Calculate the angle with the horizontal axis
                            angle = np.arctan(slope) * (180 / np.pi)  # Convert to degrees

                            # Generate x values for the polynomial line
                            x_line = np.linspace(x_coords.min(), x_coords.max(), 100)
                            y_line = np.polyval(coeffs, x_line)

                            # Draw the polynomial regression line
                            for j in range(len(x_line) - 1):
                                start_point = (int(x_line[j]), int(y_line[j]))
                                end_point = (int(x_line[j + 1]), int(y_line[j + 1]))
                                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                            # Escreve o ângulo ao lado da linha
                            angle_text = f"Angle: {angle:.2f}°"  # Adiciona o símbolo de grau
                            text_position = (int(x_line[0]), int(y_line[0]))  # Posição para exibir o texto
                            cv2.putText(frame, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Create a unique identifier for each instance
                            obj_id = f"{class_name}_{i}"  # Unique ID for each instance
                            detected_objects_info.append(f"Object ID: {obj_id} | Angle of Regression Line: {angle:.2f} degrees")

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

        # Display all detected object information after processing
        detected_objects_placeholder.write("\n".join(detected_objects_info))

    elif source_type == 'Upload Image' and image_source is not None:
        image = cv2.imread(image_source)
        results = model(image, conf=confidence_threshold, iou=nms_threshold)

        for result in results:
            masks = result.masks  # Get segmentation masks
            if masks is not None:
                for i in range(len(masks)):
                    class_id = int(result.boxes[i].cls[0])
                    class_name = model.names[class_id]

                    # Get the mask points directly
                    mask_points = masks.xy[i]  # NumPy array
                    if mask_points.size > 0:
                        # Draw contours based on the mask points
                        contour = mask_points.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, contour, isClosed=True, color=(255, 0, 0), thickness=2)

                        # Calculate the coordinates for regression
                        x_coords = mask_points[:, 0]
                        y_coords = mask_points[:, 1]

                        # Fit a polynomial (degree 1 for a linear fit)
                        coeffs = np.polyfit(x_coords, y_coords, 1)  # Degree 1 for linear
                        slope = coeffs[0]  # Coefficient for x (slope)

                        # Calculate the angle with the horizontal axis
                        angle = np.arctan(slope) * (180 / np.pi)  # Convert to degrees

                        # Generate x values for the polynomial line
                        x_line = np.linspace(x_coords.min(), x_coords.max(), 100)
                        y_line = np.polyval(coeffs, x_line)

                        # Draw the polynomial regression line
                        for j in range(len(x_line) - 1):
                            start_point = (int(x_line[j]), int(y_line[j]))
                            end_point = (int(x_line[j + 1]), int(y_line[j + 1]))
                            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

                        # Escreve o ângulo ao lado da linha
                        angle_text = f"Angle: {angle:.2f}°"  # Adiciona o símbolo de grau
                        text_position = (int(x_line[0]), int(y_line[0]))  # Posição para exibir o texto
                        cv2.putText(image, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Create a unique identifier for each instance
                        obj_id = f"{class_name}_{i}"  # Unique ID for each instance
                        detected_objects_info.append(f"Object ID: {obj_id} | Angle of Regression Line: {angle:.2f} degrees")

        st.image(image, channels="BGR", use_column_width=True)

        # Display all detected object information after processing
        detected_objects_placeholder.write("\n".join(detected_objects_info))
