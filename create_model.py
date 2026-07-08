import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create a simple dummy model for testing
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a simple sequential model
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation='softmax')  # 2 classes: Cat and Dog
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save('image_classifier.keras')
print("✓ Model file created successfully!")
