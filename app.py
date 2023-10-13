import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

app = Flask(__name)


# Constants
DATA_DIR = "/content/drive/MyDrive/Speech_Accent/recordings"
MAX_FEATURE_LENGTH = 2000  # Maximum length for padding/truncating

# Function to extract MFCC features from audio file
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)

    # Pad or truncate to make consistent length
    if mfccs.shape[1] < MAX_FEATURE_LENGTH:
        pad_width = MAX_FEATURE_LENGTH - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_FEATURE_LENGTH]

    return mfccs.flatten()

# Load data and extract features
def load_data():
    features = []
    labels = []

    for subdir, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(subdir, file)
            label = os.path.basename(subdir)  # Use the immediate directory name as the label

            if file.endswith('.mp3'):
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(label)

    return np.array(features), np.array(labels)

# Load and preprocess data
features, labels = load_data()

# Ensure features and labels are non-empty
if len(features) == 0:
    print("No features extracted. Check data directory and file formats.")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Provide feedback based on accuracy
if accuracy >= 0.75:
    print('Excellent')
elif accuracy >= 0.5:
    print('Very Good')
else:
    print('49% or less. Try again.')
# Define your audio processing code here (adapt the provided code).

@app.route("/match", methods=["POST"])
def match_cars_and_speed():
    data = request.get_json()
    car_name = data.get("carName")
    speed = data.get("speed")

    # Process the car name and speed using your audio processing code.
    # Replace this with the actual processing logic.
    result = f"Matching {car_name} with a speed of {speed}..."

    return jsonify({"result": result})

if __name__ == "__main":
    app.run()
