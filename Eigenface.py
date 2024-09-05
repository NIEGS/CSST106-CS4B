import cv2
import numpy as np
import os

# Function to read images from a folder and assign labels
def read_images(data_folder_path):
    images = []  # List to store the images
    labels = []  # List to store the corresponding labels
    label = 0    # Initial label (0 for the first person)
    
    # List all items in the data folder path
    people = os.listdir(data_folder_path)

    for person in people:
        person_folder_path = os.path.join(data_folder_path, person)  # Full path to person's folder
        
        # Check if the path is a directory
        if os.path.isdir(person_folder_path):
            for image_filename in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_filename)  # Full path to the image
                
                # Check if the path is a file
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                    if image is not None:
                        images.append(np.asarray(image, dtype=np.uint8))  # Convert image to numpy array and add to list
                        labels.append(label)  # Assign the current label to the image
                    else:
                        print(f"Warning: Could not read image {image_path}")
            label += 1  # Increment label for the next person

    return [images, np.asarray(labels, dtype=np.int32)]  # Return images and labels as numpy arrays

# Path to the folder containing training images (replace with your actual path)
data_folder_path = 'Original Images'

# Load training data by reading images and labels from the specified folder
[training_images, labels] = read_images(data_folder_path)

# Create the Eigenface Recognizer model using OpenCV
model = cv2.face.EigenFaceRecognizer_create()

# Train the model using the training images and their labels
model.train(np.asarray(training_images), np.asarray(labels))

# Function to predict the label of a test image
def predict(test_image_path):
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)  # Read the test image in grayscale
    label, confidence = model.predict(test_image)  # Predict the label and get the confidence score
    return label, confidence  # Return the predicted label and confidence

# Path to the test image (replace with your actual path)
test_image_path = 'Original Images/Brad Pitt'

# Predict the face in the test image using the trained model
predicted_label, confidence = predict(test_image_path)

# Output the result: predicted label (person) and the confidence score
print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")
