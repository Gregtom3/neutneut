import os
import numpy as np
import h5py
from tqdm import tqdm
import argparse

def load_unzip_data(h5file):
    """Function to load data from the h5 file."""
    with h5py.File(h5file, 'r') as f:
        x = np.array(f['X'])  # assuming your data is stored under the key 'X'
        y = np.array(f['y'])  # assuming your data is stored under the key 'y'
        m = np.array(f['misc'])  # assuming your data is stored under the key 'misc'
    return x, y, m

def save_updated_data_to_h5(h5file, x, y, m):
    """Function to save updated data to the h5 file."""
    with h5py.File(h5file, 'w') as f:
        f.create_dataset('X', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('misc', data=m)

def process_files(project_directory):
    # List of all h5 files in the project directory
    h5files = [f"{project_directory}/{file}" for file in sorted(os.listdir(project_directory)) if file.endswith(".h5")]

    # Initialize the shift value for y (starts at 0 for the first file)
    max_y_value = 0

    # Loop over each h5 file
    for h5file in tqdm(h5files, desc="Processing files"):
        # Load data from the current h5 file
        x, y, m = load_unzip_data(h5file)

        # Only update y values that are >= 0
        y_mask = y >= 0

        # Shift the y values in the current file
        y[y_mask] += max_y_value

        # Update the max_y_value for the next file
        max_y_value = np.max(y[y_mask]) if np.any(y_mask) else max_y_value
        max_y_value += 1

        # Save the updated data back to the h5 file
        save_updated_data_to_h5(h5file, x, y, m)

def main():
    parser = argparse.ArgumentParser(description="Process and update HDF5 files for a specific project.")
    parser.add_argument('--project', type=str, required=True, help='Project dir')
    args = parser.parse_args()

    # Base directory
    base_directory = '../'

    # Full project directory
    project_directory = os.path.join(base_directory, args.project, 'training')

    # Process the files in the specified project directory
    process_files(project_directory)

if __name__ == "__main__":
    main()
