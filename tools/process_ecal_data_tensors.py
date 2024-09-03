import sys
import os
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from TrainData import TrainData

def preprocess_into_tensors(csv_filename, output_h5_filename, data_type, train_test_split=0.8):
    if data_type not in ["mc", "rec"]:
        print("Error: data_type must be either 'mc' or 'rec'.")
        sys.exit(1)
    
    # If "rec" is chosen, set train_test_split to 1.0
    if data_type == "rec":
        train_test_split = 1.0
    
    # Preprocess into tensors
    data = TrainData(csv_files=[csv_filename],
                     train_size=train_test_split,
                     return_tensor=True,
                     K=100,
                     remove_background=True,
                     min_particles=1)

    train_X, train_y, train_misc = data.get_train_data(-1)
    
    # Determine whether to split into train/test or save as a single dataset
    if train_test_split < 1.0:
        output_h5_filename_train = output_h5_filename.replace('.h5', '_train.h5')
        output_h5_filename_test = output_h5_filename.replace('.h5', '_test.h5')

        with h5py.File(output_h5_filename_train, 'w') as hf:
            hf.create_dataset('train_X', data=train_X)
            hf.create_dataset('train_y', data=train_y)
            hf.create_dataset('train_misc', data=train_misc)
        print(f"Training data saved to {output_h5_filename_train}")
        
        test_X, test_y, test_misc = data.get_test_data(-1)
        with h5py.File(output_h5_filename_test, 'w') as hf:
            hf.create_dataset('test_X', data=test_X)
            hf.create_dataset('test_y', data=test_y)
            hf.create_dataset('test_misc', data=test_misc)
        print(f"Testing data saved to {output_h5_filename_test}")
    
    else:
        # If "rec", just save as a single dataset
        with h5py.File(output_h5_filename, 'w') as hf:
            hf.create_dataset('train_X', data=train_X)
            hf.create_dataset('train_y', data=train_y)
            hf.create_dataset('train_misc', data=train_misc)
        print(f"Dataset saved to {output_h5_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python preprocess_to_tensors.py <input_csv_filename> <output_h5_filename> <data_type: mc/rec> <train_test_split>")
        sys.exit(1)
    
    csv_filename = sys.argv[1]
    output_h5_filename = sys.argv[2]
    data_type = sys.argv[3]
    if len(sys.argv) == 4:
        train_test_split = 0.8
    else:
        train_test_split = float(sys.argv[4])

    preprocess_into_tensors(csv_filename, output_h5_filename, data_type, train_test_split)