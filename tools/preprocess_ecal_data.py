import sys
import os

# Add the relative path to the src directory where ECALDataAnalyzer is located
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from ECALDataAnalyzer import ECALDataAnalyzer

def main():
    if len(sys.argv) != 3:
        print("Usage: python preprocess_ecal_data.py <input_hipo_filename> <output_csv_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # Initialize the ECALDataAnalyzer with the provided filenames
    ana = ECALDataAnalyzer(input_filename=input_filename, output_filename=output_filename)

    # Process the HIPO file and generate the output CSV
    ana.process_hipo()

if __name__ == "__main__":
    main()
