import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, classification_report
from tensorflow.keras.preprocessing import image
import joblib

#Hem creat un model per el dataset de números, un altre per el dataset de lletres i un general.

data_dir = r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\datasetaina'
img_size = (128, 64)

def load_images_and_labels(data_dir, img_size):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = image.load_img(img_path, target_size=img_size)
                img_array = image.img_to_array(img)
                img_array = img_array.flatten()  
                images.append(img_array)
                labels.append(label)  
    return np.array(images), np.array(labels)


X, y = load_images_and_labels(data_dir, img_size)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_val)
precision = precision_score(y_val, y_pred, average='weighted')
print(f'Precisió en el conjunt de validació: {precision * 100:.2f}%')
joblib.dump(knn_model, 'model_KNN_general.pkl')
print("Model guardat amb èxit!")
