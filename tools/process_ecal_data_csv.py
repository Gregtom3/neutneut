import sys
import os
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from ECALDataAnalyzer import ECALDataAnalyzer

def main(input_filename, output_csv_filename, data_type):
    # Initialize the ECALDataAnalyzer with the provided filenames and data type
    ana = ECALDataAnalyzer(input_filename=input_filename, output_filename=output_csv_filename, data_type=data_type)

    # Process the HIPO file and generate the output CSV
    ana.process_hipo()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_ecal_data_file.py <input_hipo_filename> <output_csv_filename> <data_type: mc/rec>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_csv_filename = sys.argv[2]
    data_type = sys.argv[3]

    main(input_filename, output_csv_filename, data_type)