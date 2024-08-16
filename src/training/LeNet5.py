import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import processing.extract as extract
from models.LeNet import LeNet5

train_images = extract.read_idx('data/raw/train-images-idx3-ubyte/train-images.idx3-ubyte')
train_labels = extract.read_idx('data/raw/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images = extract.read_idx('data/raw/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels = extract.read_idx('data/raw/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

# Add channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_images, test_images = train_images / 255.0, test_images / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# Instantiate the model
model = LeNet5()

# Compile the model
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc:.4f}')

# # Save the model
# model.save('lenet5_mnist_model.h5')
