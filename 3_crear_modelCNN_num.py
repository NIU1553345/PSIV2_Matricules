import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.metrics import Precision

# La carpeta ha de tenir 10 carpetes (0,1,2...9) amb imatges del número corresponent 
data_dir = r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\Naina'

train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Inferir etiquetes de las carpetes (0-9)
    label_mode='categorical',  # One-hot encoding 
    image_size=(128, 64),  
    batch_size=32,
    validation_split=0.2,  
    subset='training',
    seed=123
)

val_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(128, 64),
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Definir arquitectura de la CNN
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluar el model
test_loss, test_precision = model.evaluate(val_dataset)
print(f'Precisió en el conjunt de validació: {test_precision * 100:.2f}%')

model.save('model_CNN_num.h5')

