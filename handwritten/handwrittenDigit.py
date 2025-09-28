import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Pick an index
i = 24

# Plot image
plt.figure(figsize=(4,4))   # <-- ensures window size
plt.imshow(X_test[i], cmap='gray')
plt.title(f"Actual: {y_test[i]}")
plt.axis("off")             # <-- hides axis
plt.show()

# Predict
pred = model.predict(X_test[i:i+1])
print(f"Predicted: {pred.argmax()}")
