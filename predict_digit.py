import cv2
import numpy as np
import pickle
import os

# Load trained model
with open("digits_model.pkl", "rb") as file:
    model = pickle.load(file)

# Folder containing images
image_folder = "test_images"

# Supported image extensions
extensions = (".png", ".jpg", ".jpeg", ".bmp")

# Get all images
images = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith(extensions)
]

if len(images) == 0:
    print("No images found in test_images folder.")
    exit()

print("\nDigit Prediction")
print("-" * 30)

for filename in images:

    image_path = os.path.join(image_folder, filename)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 8x8
    resized = cv2.resize(gray, (8, 8))

    # Invert image
    resized = 255 - resized

    # Convert pixel values from 0-255 to 0-16
    resized = resized / 16.0

    # Convert to one-dimensional array
    data = resized.reshape(1, -1)

    # Predict
    prediction = model.predict(data)

    print(f"{filename} --> Predicted Digit: {prediction[0]}")

    # Show image
    cv2.imshow("Input Image", image)

    # Press any key for next image
    cv2.waitKey(0)

cv2.destroyAllWindows()