import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# La carpeta ha de tenir 20 carpetes (B,C,D...Z) amb imatges de la lletra corresponent 
data_dir = r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\LLaina'

# Crear datasets de entrenamiento y validación
train_dataset = image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Inferir etiquetas de las carpetes
    label_mode='categorical',  # Utilitzar one-hot encoding per les etiquetes
    image_size=(128, 64),  # Redimensionar les imatges a 64x64
    batch_size=32,
    validation_split=0.2,  # Utilitzar 20% per la validació
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

# Definir l'arquitectura de la CNN
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
    layers.Dense(20, activation='softmax')  
])

# Compilar el model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=6) #El número de èpoques es pot modificar

# Evaluar el model
test_loss, test_precision = model.evaluate(val_dataset)
print(f'Precisió en el conjunt de validació: {test_precision * 100:.2f}%')

model.save('model_CNN_lletres.h5')
