import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model for prediction
model = load_model(r'D:\Python programs\SIH\models\crop_disease_model_corn1.keras')

# Define the class names (ensure they match the order in which they were trained)
class_names = [
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight',
    'Corn_(maize)_healthy'
]

# Define the disease information (Preventive Measures and Medications)
disease_info = {
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': {
        'Preventive Measures': "Practice crop rotation and use resistant varieties. Ensure good field drainage.",
        'Medications': "Apply fungicides like chlorothalonil or azoxystrobin to manage symptoms."
    },
    'Corn_(maize)Common_rust': {
        'Preventive Measures': "Use resistant varieties and practice crop rotation. Avoid planting in the same field year after year.",
        'Medications': "Fungicides such as propiconazole or tebuconazole can help control rust."
    },
    'Corn_(maize)Northern_Leaf_Blight': {
        'Preventive Measures': "Practice crop rotation and remove infected plant debris. Ensure good air circulation.",
        'Medications': "Apply fungicides like mancozeb or pyraclostrobin to control the disease."
    },
    'Corn_(maize)_healthy': {
        'Preventive Measures': "Maintain good field management practices including proper irrigation and pest control.",
        'Medications': "No action needed."
    },
}

# Load and preprocess the test image
img_path = r'D:\Python programs\SIH\train\train\corn final\Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot\6f4b40eb-6d14-4f30-9ef8-89f85bd841fd__RS_GLSp 4426.JPG'  # Update with the path to the corn image
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