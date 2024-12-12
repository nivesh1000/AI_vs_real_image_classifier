import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('model')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Reads and preprocesses an image for prediction.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).
    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to a NumPy array
    img_array = img_to_array(img)
    # Rescale pixel values (assuming the model was trained on normalized data)
    img_array = img_array / 255.0
    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_image(image_path, model):
    """
    Predicts whether an image is real or AI-generated.
    Args:
        image_path (str): Path to the image file.
        model: Trained Keras model for prediction.
    Returns:
        str: Prediction result.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make the prediction
    prediction = model.predict(preprocessed_image)
    print(prediction[0][0])
    # Decode the result
    class_label = "Real" if prediction[0][0] > 0.5 else "AI-Generated"
    confidence = prediction[0][0] if class_label == "Real" else 1 - prediction[0][0]
    return f"Prediction: {class_label} with confidence {confidence:.2f}"

# Example usage
image_path = '/home/cynoteckdell/Documents/RealvsAIgen-Face-Classifier/Dataset/photowhite.jpeg'  # Replace with the path to your image
result = predict_image(image_path, model)
print(result)
