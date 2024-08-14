import tensorflow as tf

def build_functional_model():
    # Define the input layer
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Convolutional layer: 6 filters, 5x5 kernel, activation function
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='tanh')(inputs)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    # Convolutional layer: 16 filters, 5x5 kernel, activation function
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    # Flatten the output of the last convolutional layer
    x = tf.keras.layers.Flatten()(x)

    # Fully connected layer: 120 units, activation function
    x = tf.keras.layers.Dense(units=120, activation='tanh')(x)

    # Fully connected layer: 84 units, activation function
    x = tf.keras.layers.Dense(units=84, activation='tanh')(x)

    # Output layer: 10 units for classification (10 classes for MNIST dataset)
    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def build_sequential_model():
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

    return model

class LeNet5(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='tanh')
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh')
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=120, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(units=84, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)

if __name__ == "__main__":
    # Build and summarize the Functional API model
    functional_model = build_functional_model()
    print("Functional API Model Summary:")
    functional_model.summary()

    # Build and summarize the Sequential model
    sequential_model = build_sequential_model()
    print("\nSequential API Model Summary:")
    sequential_model.summary()

    # Build and summarize the custom LeNet5 model
    lenet5_model = LeNet5()
    print("\nCustom LeNet5 Model Summary:")
    lenet5_model.build((None, 28, 28, 1))  # Build the model to get the summary
    lenet5_model(tf.zeros((1, 28, 28, 1)))
    lenet5_model.summary()