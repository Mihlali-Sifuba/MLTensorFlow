import numpy as np

def read_idx(filename: str)-> np.ndarray:
    """
    Reads data from an IDX file (image or label file format).
    
    Parameters:
        filename (str): Path to the IDX file.
    
    Returns:
        np.ndarray: Data read from the IDX file. 
                    For image files, it returns a 3D array (num_items, num_rows, num_cols).
                    For label files, it returns a 1D array (num_items,).
    
    Raises:
        ValueError: If the magic number is unknown or if file format is invalid.
        FileNotFoundError: If the file does not exist.
        IOError: If there is an issue reading the file.
    """
    try:
        with open(filename, 'rb') as f:
            # Read the magic number
            magic_number = int.from_bytes(f.read(4), 'big')
            
            # Read the number of items (images or labels)
            num_items = int.from_bytes(f.read(4), 'big')
            
            # For images, read the number of rows and columns
            if magic_number == 2051:  # Images file magic number
                num_rows = int.from_bytes(f.read(4), 'big')
                num_cols = int.from_bytes(f.read(4), 'big')
                # Read the image data
                data = np.fromfile(f, dtype=np.uint8).reshape(num_items, num_rows, num_cols)
                # Check if the file size matches the expected number of bytes
                expected_size = num_items * num_rows * num_cols
                actual_size = data.size
                if actual_size != expected_size:
                    raise ValueError("File size does not match expected number of items.")
            elif magic_number == 2049:  # Labels file magic number
                # Read the label data
                data = np.fromfile(f, dtype=np.uint8)
            else:
                raise ValueError("Unknown IDX magic number: {}".format(magic_number))
        
        return data
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filename}") from e
    except IOError as e:
        raise IOError(f"Error reading file: {filename}") from e
    except ValueError as e:
        raise ValueError(f"Value error: {e}") from e

if __name__ == "__main__":
    try:
        # Example usage
        train_images = read_idx('data/raw/train-images-idx3-ubyte/train-images.idx3-ubyte')
        train_labels = read_idx('data/raw/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
        test_images = read_idx('data/raw/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
        test_labels = read_idx('data/raw/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

        print(f'Train Images Shape: {train_images.shape}')  # Should print (number_of_images, 28, 28)
        print(f'Train Labels Shape: {train_labels.shape}')  # Should print (number_of_labels,)
        print(f'Test Images Shape: {test_images.shape}')  # Should print (number_of_images, 28, 28)
        print(f'Test Labels Shape: {test_labels.shape}')  # Should print (number_of_labels,)
    except Exception as e:
        print(f"An error occurred: {e}")