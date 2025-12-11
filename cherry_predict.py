import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model for prediction
model = load_model(r'D:\Python programs\SIH\models\crop_disease_model_cherry.keras')

# Define the class names (ensure they match the order in which they were trained)
class_names = [
    'Cherry_(including_sour)_Powdery_mildew',
    'Cherry_(including_sour)_healthy'
]

# Define the disease information (Preventive Measures and Medications)
disease_info = {
    'Cherry_(including_sour)_Powdery_mildew': {
        'Preventive Measures': "Ensure good air circulation and avoid overhead watering. Remove and destroy infected plant parts.",
        'Medications': "Apply fungicides such as sulfur or potassium bicarbonate to control the spread."
    },
    'Cherry_(including_sour)_healthy': {
        'Preventive Measures': "Maintain proper orchard management practices including regular inspections and balanced fertilization.",
        'Medications': "No action needed."
    },
}

# Load and preprocess the test image
img_path = r'D:\Python programs\SIH\train\train\cherry_final\Cherry_(including_sour)__Powdery_mildew\2f2fdc52-8420-41d1-ab72-320a99c27537__FREC_Pwd.M 5101.JPG'  # Update with the path to the cherry image
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