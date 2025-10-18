# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the Fashion MNIST dataset (contains 70,000 grayscale images of clothing items)
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

# Display one sample image and its label
plt.imshow(train_images[5])
plt.title(f"Label: {train_labels[5]}")
plt.show()

# Class names corresponding to labels (0–9)
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize pixel values (0–255 → 0–1) to improve model performance
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build an Artificial Neural Network (ANN) model
model = Sequential()

# Flatten the 28x28 input images into a 1D vector
model.add(Flatten(input_shape=(28, 28)))

# Add a hidden layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Output layer with 10 units (for 10 clothing categories)
model.add(Dense(10))

# Compile the model with loss, optimizer, and metric
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# Show model architecture summary
model.summary()

# Train the model for 10 epochs on the training data
h = model.fit(train_images, train_labels, epochs=10)

# Make predictions on test data
y_pred = model.predict(test_images)

# Evaluate model performance
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.3f}")

# Example: Predict the label for a custom image
test = cv2.imread('boot2.png')  # Load image
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
test = cv2.resize(test, (28, 28))  # Resize to 28x28 to match dataset format
test = test / 255.0  # Normalize pixel values
test = np.array([test])  # Convert to batch format (1, 28, 28)

# Predict the class
out = model.predict(test)
predicted_label = class_names[np.argmax(out)]

print("Predicted Outfit:", predicted_label)
