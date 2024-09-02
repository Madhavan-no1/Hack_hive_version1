import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_images_from_folder(folder):
    images = []
    labels = []
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            img = cv2.resize(img, (128, 128))
            img = img.flatten()
            images.append(img)
            labels.append(category)
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_images_from_folder(r"C:\Users\Mazveen\project\dataset")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train a new SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the new model
joblib.dump(model, 'new_svm_model.pkl')

# Evaluate the new model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
