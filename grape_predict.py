import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model for prediction
model = load_model(r'D:\Python programs\SIH\models\crop_disease_model_grape.keras')

# Define the class names (ensure they match the order in which they were trained)
class_names = [
               'Grape_Black_rot', 'Grape_Esca(Black_Measels)', 'Grapeleaf_blight(Isariopsis_leaf_spot)', 'Grape_healthy',
               ]

# Define the disease information (Preventive Measures and Medications)
disease_info = {
    'Grape__Black_rot': {
        'Preventive Measures': "Prune vines to improve airflow and remove diseased plant debris. Rotate crops regularly.",
        'Medications': "Apply fungicides such as mancozeb or captan to control the spread."
    },
    'Grape_Esca(Black_Measels)': {
        'Preventive Measures': "Avoid injuries to vines during pruning and limit water stress.",
        'Medications': "There are no effective chemical treatments for Esca. Manage the disease through good viticultural practices."
    },
    'Grape__leaf_blight(Isariopsis_leaf_spot)': {
        'Preventive Measures': "Improve air circulation by pruning and avoid overwatering. Remove infected leaves.",
        'Medications': "Fungicide applications, such as copper-based products, may help."
    },
    'Grape__healthy': {
        'Preventive Measures': "Maintain regular inspection of plants and ensure proper irrigation practices.",
        'Medications': "No action needed."
    },
}

# Load and preprocess the test image
img_path = r'D:\Python programs\SIH\PlantVillage\train\Grape__Black_rot\0e7726c0-a309-4194-b2e6-d0e33af39373__FAM_B.Rot 0530.JPG'  # Make sure to provide the full path to the image
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Resize the image to match the input size of the model (128x128)
img_resized = cv.resize(img, (180, 180))

# Normalize the image
img_array = np.array([img_resized]) / 255.0  # Scale pixel values to [0, 1]

# Display the image
plt.imshow(img_resized)
plt.axis('off')  # Hide the axes for a better display
plt.show()

# Make the prediction
prediction = model.predict(img_array)
index = np.argmax(prediction)

# Get the predicted label
predicted_label = class_names[index]
print(f'Prediction: {predicted_label}')

# Fetch and display preventive measures and medications
if predicted_label in disease_info:
    preventive_measures = disease_info[predicted_label]['Preventive Measures']
    medications = disease_info[predicted_label]['Medications']

    print(f"Preventive Measures: {preventive_measures}")
    print(f"Medications: {medications}")
else:
    print("No information available for this disease.")