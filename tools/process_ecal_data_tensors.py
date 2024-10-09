import sys
import os
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from TrainData import TrainData
from global_params import *

def preprocess_into_tensors(csv_filename, output_h5_filename):

    # Preprocess into tensors
    data = TrainData(csv_files=[csv_filename],
                     K=K)

    X, y, misc = data.get_data()

    with h5py.File(output_h5_filename, 'w') as hf:
        hf.create_dataset('X', data=X)
        hf.create_dataset('y', data=y)
        hf.create_dataset('misc', data=misc)
    print(f"Dataset saved to {output_h5_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_to_tensors.py <input_csv_filename> <output_h5_filename>")
        sys.exit(1)
    
    csv_filename = sys.argv[1]
    output_h5_filename = sys.argv[2]

    preprocess_into_tensors(csv_filename, output_h5_filename)