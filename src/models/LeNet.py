import tensorflow as tf


# Create a Sequential model
model = tf.keras.Sequential()

# Add layers according to the LeNet-5 architecture
# Input layer
model.add(tf.keras.Input(shape=(28, 28, 1)))

# Convolutional layer: 6 filters, 5x5 kernel, activation function
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='tanh'))

# Average pooling layer
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2))

# Convolutional layer: 16 filters, 5x5 kernel, activation function
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))

# Average pooling layer
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2))

# Flatten the output of the last convolutional layer
model.add(tf.keras.layers.Flatten())

# Fully connected layer: 120 units, activation function
model.add(tf.keras.layers.Dense(units=120, activation='tanh'))

# Fully connected layer: 84 units, activation function
model.add(tf.keras.layers.Dense(units=84, activation='tanh'))

# Output layer: 10 units for classification (10 classes for MNIST dataset)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Print the model summary
model.summary()
