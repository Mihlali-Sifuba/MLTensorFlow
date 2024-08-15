import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from processing.extract import read_idx

def plot_images(images: np.ndarray, labels: np.ndarray, num_images: int = 10):
    """
    Visualizes a subset of images from the dataset.

    Parameters:
        images (np.ndarray): Array of images to visualize.
        labels (np.ndarray): Array of labels corresponding to the images.
        num_images (int): Number of images to visualize.
    """
    if num_images > len(images):
        num_images = len(images)

    # Create a grid for the plots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')  # Hide the axis
    
    plt.show()

if __name__ == "__main__":
    try:
        # Load data
        train_images = read_idx('data/raw/train-images-idx3-ubyte/train-images.idx3-ubyte')
        train_labels = read_idx('data/raw/train-labels-idx1-ubyte/train-labels.idx1-ubyte')

        # Visualize a subset of images and their labels
        plot_images(train_images, train_labels, num_images=10)
    
    except Exception as e:
        print(f"An error occurred: {e}")

