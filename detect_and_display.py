import cv2
import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
# Path to your trained model. Make sure this file is in the same directory or provide the full path.
MODEL_PATH = 'rps_hand_detector.h5'
IMAGE_SIZE = (128, 128)  # Must match the size used during training

# IMPORTANT: Map the class indices from training to readable labels.
# This dictionary MUST exactly match the output from the 'Data Loading and Preprocessing' cell in your Jupyter Notebook.
# Example: If your notebook output was {'paper': 0, 'rock': 1, 'scissors': 2}, then use:
CLASS_LABELS = {0: 'paper', 1: 'rock', 2: 'scissors'} # Adjust this based on your actual labels!

# --- Load Trained Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please ensure you have run the Jupyter Notebook to train and save the model, and that the .h5 file is in the correct location.")
    exit()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file is not corrupted and is a valid Keras model.")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # 0 for the default webcam. Change if you have multiple cameras.

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please check if a webcam is connected and not in use by another application.")
    exit()

print("Webcam initialized. Place your hand within the green box. Press 'q' to quit.")

# --- Real-time Detection Loop ---
while True:
    ret, frame = cap.read() # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a more natural mirror view
    frame = cv2.flip(frame, 1)

    # --- Define Region of Interest (ROI) for the hand ---
    # We'll define a square region in the center of the frame
    # The user should place their hand inside this box.
    frame_height, frame_width, _ = frame.shape
    roi_size = min(frame_height, frame_width) // 2 # Take half the smaller dimension for ROI size
    x1 = (frame_width - roi_size) // 2
    y1 = (frame_height - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # Draw the ROI box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box, 2 pixels thick

    # Crop the ROI from the frame
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI for prediction
    # 1. Resize to the target size (e.g., 128x128)
    img_for_prediction = cv2.resize(roi, IMAGE_SIZE)
    
    # 2. Add a batch dimension (model expects a batch of images: (1, height, width, channels))
    img_for_prediction = np.expand_dims(img_for_prediction, axis=0)
    # 3. Normalize pixel values (0-255 to 0-1), same as during training
    img_for_prediction = img_for_prediction / 255.0

    # --- Make Prediction ---
    predictions = model.predict(img_for_prediction, verbose=0) # verbose=0 suppresses progress bar
    predicted_class_index = np.argmax(predictions) # Get the index of the class with highest probability
    confidence = np.max(predictions) * 100        # Get the confidence percentage

    # Get the human-readable label
    predicted_label = CLASS_LABELS.get(predicted_class_index, "Unknown")

    # --- Display Results on the frame ---
    text = f"{predicted_label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Red text

    # Display the processed frame
    cv2.imshow('Rock Paper Scissors Detector', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()          # Release the webcam
cv2.destroyAllWindows() # Close all OpenCV windows