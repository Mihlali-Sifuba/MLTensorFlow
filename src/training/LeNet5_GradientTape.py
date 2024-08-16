import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import processing.extract as extract
from models.LeNet import LeNet5

# Load and preprocess the data
train_images = extract.read_idx('data/raw/train-images-idx3-ubyte/train-images.idx3-ubyte')
train_labels = extract.read_idx('data/raw/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images = extract.read_idx('data/raw/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels = extract.read_idx('data/raw/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

# Add channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Normalize images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# Instantiate the model
model = LeNet5()

# Define the loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Training step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

# Validation step
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)

    test_loss.update_state(loss)
    test_accuracy.update_state(labels, predictions)

# Training loop
EPOCHS = 10

for epoch in range(EPOCHS):
    # Reset metrics at the start of each epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    # Training
    for images, labels in train_dataset:
        train_step(images, labels)

    # Testing
    for images, labels in test_dataset:
        test_step(images, labels)

    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss.result():.4f}, Train Accuracy: {train_accuracy.result():.4f}')
    print(f'Test Loss: {test_loss.result():.4f}, Test Accuracy: {test_accuracy.result():.4f}')

# Save the model
model.save('lenet5_mnist_model.h5')
