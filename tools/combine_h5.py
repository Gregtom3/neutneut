import os
import sys
import glob
import h5py

def combine_h5_files(directory, extension):
    h5_files = glob.glob(os.path.join(directory, f'*{extension}.h5'))
    if not h5_files:
        print(f"No {extension} .h5 files found in the directory.")
        return
    combined_file = os.path.join(directory, f'dataset{extension}.h5')
    
    with h5py.File(combined_file, 'w') as h5_out:
        total_lengths = {}  # Dictionary to store the total length of each dataset
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as h5_in:
                    for key in h5_in.keys():
                        if key in h5_out:
                            h5_out[key].resize((h5_out[key].shape[0] + h5_in[key].shape[0]), axis=0)
                            h5_out[key][-h5_in[key].shape[0]:] = h5_in[key][:]
                            total_lengths[key] += h5_in[key].shape[0]  # Update total length
                        else:
                            maxshape = (None,) + h5_in[key].shape[1:]  # Set the first dimension as unlimited
                            chunks = h5_in[key].chunks  # Use the same chunk size as the input dataset
                            h5_out.create_dataset(key, data=h5_in[key], maxshape=maxshape, chunks=chunks)
                            total_lengths[key] = h5_in[key].shape[0]  # Initialize total length
            except (OSError, IOError) as e:
                print(f"Skipping file {h5_file} due to error: {e}")

        # Store the total sizes as attributes in the combined HDF5 file
        for key, length in total_lengths.items():
            h5_out[key].attrs['total_length'] = length  # Save the total length as an attribute

    # Delete original .h5 files after combining, except the ones that match the extensions
#     for h5_file in h5_files:
#         if os.path.basename(h5_file) != os.path.basename(combined_file) and extension in h5_file:
#             os.remove(h5_file)
#     print(f"Combined {len(h5_files)} files into {combined_file} and deleted originals.")
    print(f"Combined {len(h5_files)} files into {combined_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: combine_h5.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    # Combine training files
    combine_h5_files(directory, '_train')

    # Combine testing files
    combine_h5_files(directory, '_test')