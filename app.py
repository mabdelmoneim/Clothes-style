import cv2
from sklearn.cluster import KMeans
import webcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from numpy.linalg import norm
import plotly.express as px
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Sequential
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm
import pickle


app = Flask(__name__)




# Load  Model
with open("C:\\Users\\Mohamed\\Desktop\\mypythonapp\\size.pkl", "rb") as f:
    model_1 = pickle.load(f)



def extract_colors(image, num_colors=5):
  # Convert image from BGR to RGB
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Flatten the image to a 2D array of pixels
  pixels = image.reshape(-1, 3)

  # Perform color clustering using K-means
  kmeans = KMeans(n_clusters=num_colors)
  kmeans.fit(pixels)

  # Get the RGB values of the cluster centers
  colors_rgb = kmeans.cluster_centers_

  # Convert the RGB values to integers
  colors_rgb = colors_rgb.astype(int)

  # Return the recommended colors as RGB values
  return colors_rgb

def map_rgb_to_color_name(rgb):
  differences = {}
  for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
    r, g, b = webcolors.hex_to_rgb(color_hex)
    differences[sum([(r - rgb[0]) ** 2,
                      (g - rgb[1]) ** 2,
                      (b - rgb[2]) ** 2])] = color_name
  return differences[min(differences.keys())]

def select_unique_preferred_colors(colors):
  preferred_colors = set()
  unique_colors = []

  for color in colors:
    rgb = tuple(color.tolist())
    color_name = map_rgb_to_color_name(rgb)

    # Check if color is unique (not in set or color name list)
    if rgb not in preferred_colors and color_name not in [name for _, name in unique_colors]:
      unique_colors.append((rgb, color_name))
      preferred_colors.add(rgb)

    # Enforce selection of 5 unique colors
    if len(unique_colors) == 5:
      break

  return unique_colors

def display_colors(colors):
  num_colors = len(colors)
  fig, ax = plt.subplots(1, num_colors, figsize=(num_colors * 2, 2))

  for i, (rgb, color_name) in enumerate(colors):
    color_patch = np.ones((10, 10, 3), dtype=np.uint8) * rgb
    ax[i].imshow(color_patch)
    ax[i].axis("off")
    ax[i].set_title(color_name, fontsize=10)
    ax[i].text(0, -15, color_name, fontsize=8, ha='center')

  plt.tight_layout()
  plt.show()








# Load the ResNet50 model
preprocessing_pipeline = Pipeline([
    ('missing-value', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler(with_mean=False))
])

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a new Sequential model for feature extraction
feature_extractor = Sequential([
    model,
    GlobalMaxPooling2D()
])


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img)
    # Apply GlobalMaxPooling2D to get a feature vector of length 2048
    pooled_result = GlobalMaxPooling2D()(result)
    normalized_result = pooled_result / norm(pooled_result)
    # Ensure the feature vector is reshaped to have only one dimension
    reshaped_result = tf.reshape(normalized_result, (1, -1))
    return reshaped_result







def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices =  neighbors.kneighbors([features])
    return indices
# Create Preprocessing function

def preprocessing(image):
    img = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    return img

def classify_image(image):
    image = preprocessing(image)
    prediction = model_1.predict(image)
    score = tf.nn.softmax(prediction[0])
    return classes[tf.argmax(score)]

def classify_image(age , height, weight):
    prediction = model.predict(age, height, weight)
    return prediction

#classes = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

@app.route('/', methods=['GET'])
def index():
    return "hello"
@app.route('/api/data',methods=['GET'])
def get_data():
    # Example data to be sent to the frontend
    data = {'message': 'Hello from Flask!'}
    return jsonify(data)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    weight = float(data.get('weight'))
    age = float(data.get('age'))
    height = float(data.get('height'))

    bmi = height / weight
    weight_squared = weight * weight

    # Create a DataFrame with the user input and additional features
    user_data = pd.DataFrame({
        "weight": [weight],
        "age": [age],
        "height": [height],
        "bmi": [bmi],
        "weight_squared": [weight_squared]
    })

    # Make predictions using the loaded model
    predictions = model_1.predict(user_data)
    prediction = int(predictions)
    def transfer(prediction):
      if prediction == 1:
        return 'XS'
      elif prediction == 2:
        return 'S'
      elif prediction == 3:
        return 'M'
      elif prediction == 4:
        return 'L'
      elif prediction == 5:
        return 'XL'
      elif prediction == 6:
        return 'XXL'
      else:
        return 'XXXL'
    # Return the prediction as a JSON response
    response = {'prediction': transfer(prediction)}

    return jsonify(response)


@app.route("/color-extraction", methods=["POST"])
def color_extraction():
    # Check if an image file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files["file"]

    # Read uploaded image
    image = np.array(Image.open(io.BytesIO(file.read())).convert("RGB"))

    # Extract preferred colors
    preferred_colors = extract_preferred_colors(image)

    # Prepare response
    response = {
        "preferred_colors": preferred_colors
    }

    return jsonify(response)

@app.route("/recommend_clothes", methods=["POST"])
def recommend_clothes():
    # Check if an image file was uploaded
    if 'file' not in request.files:
         return jsonify({'error': 'No file part'})

    file = request.files['file']

    img_path = "temp.jpg"  # Save the uploaded image temporarily
    file.save(img_path)

    features = feature_extraction(img_path, model)
    print("Shape of extracted features:", features.shape)
    print("Shape of feature list:", feature_list.shape)
    indices = recommend(features[0], feature_list)

    recommended_images = [filenames[idx] for idx in indices[0]]

    return jsonify({'recommendations': recommended_images})


if __name__ == "__main__":
    app.run()
