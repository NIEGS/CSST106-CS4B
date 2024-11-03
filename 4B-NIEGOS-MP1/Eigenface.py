import cv2
import numpy as np
import os

# The read_images function reads all images from the specified directory, converts them to grayscale, and assigns labels based on the folder structure. 
#  Define Function to Read and Preprocess Images
def read_images(data_folder_path):
    images = []
    labels = []
    label = 0
    
    for person_name in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_folder_path):
            continue
        
        for image_filename in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            images.append(image)
            labels.append(label)
        
        label += 1
    
    return images, labels

# Load Training Data
data_folder_path = 'Originalimages'
training_images, labels = read_images(data_folder_path)

# Resizing Images: The images are resized to a standard size (200x200 pixels) to ensure uniformity.
# Prepare Data for Training
training_images = [cv2.resize(img, (200, 200)) for img in training_images]
training_images = np.array(training_images)
labels = np.array(labels)

# Training the Model: The Eigenface recognizer is initialized and trained using the training images and their corresponding labels.
# Initialize and Train the Eigenface Recognizer 
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, labels)

# Testing: A test image is loaded, resized, and passed to the trained model to predict the label (person's identity) and the confidence level of the prediction.
# Test the Model
test_image_path = 'Test_DwayneJohnson/Dwayne Johnson_0.jpg'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (200, 200))

label, confidence = model.predict(test_image)
print(f'Predicted Label: {label}, Confidence: {confidence}')

# This process outlines how feature extraction with Eigenfaces can address the problem of recognizing individuals with similar features and under different conditions by focusing on the most significant and discriminative features of the face.