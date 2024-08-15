import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import os
# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from processing.extract import read_idx

# Enable LaTeX for rendering
rcParams['text.usetex'] = True

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

    # Determine the number of rows and columns for the grid
    num_cols = min(num_images, 5)  # Max 5 columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division to get the number of rows

    # Create a grid for the plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: ${labels[i]}$', fontsize=14)
        axes[i].axis('off')  # Hide the axis

    # Turn off any remaining axes that are not used
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
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