import pandas as pd
import numpy as np
from itertools import combinations
from ECALDataReader import ECALDataReader
from ECALDataProcessor import ECALDataProcessor
import csv
from tqdm import tqdm

class ECALDataAnalyzer:
    """
    Class to analyze ECAL data and write the results to a CSV file
    """

    def __init__(self, input_filename, output_filename="output.csv"):
        """
        Initialize the analyzer with input and output file names.
        
        Parameters:
        -----------
        input_filename : str
            Path to the input hipo file.
        output_filename : str
            Path to the initial output CSV file for peaks (default is "output.csv").
        """
        self.input_filename = input_filename
        self.output_filename = output_filename

    def read_ecal_data_from_event(self):
        """
        Read and process ECAL data from an event file and write results to a CSV.
        """
        reader = ECALDataReader(self.input_filename)
        processor = ECALDataProcessor()
        unique_mc_counter = 0

        # Open the CSV file for writing
        with open(self.output_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            header = [
                "event_number", "mc_index", "unique_mc_index", "mc_pid", "peak_id",
                "peak_status", "peak_sector", "peak_layer", "peak_energy", "peak_time",
                "peak_xo", "peak_yo", "peak_zo", "peak_xe", "peak_ye", "peak_ze",
                "peak_width", "peak_pindex", "peak_rec_pid"
            ]
            csv_writer.writerow(header)

            # Loop over the events and process the data
            for event_number, event in tqdm(enumerate(reader.file)):
                ECAL_hits = reader.get_dict("ECAL::hits")
                ECAL_peaks = reader.get_dict("ECAL::peaks")
                ECAL_clusters = reader.get_dict("ECAL::clusters")
                REC_calorimeter = reader.get_dict("REC::Calorimeter")
                REC_particle = reader.get_dict("REC::Particle")
                MC_particle = reader.get_dict("MC::Particle")
                MC_RecMatch = reader.get_dict("MC::RecMatch")
                MC_GenMatch = reader.get_dict("MC::GenMatch")
                
                result = processor.group_hits_and_peaks_by_mcindex(
                    ECAL_hits, ECAL_clusters, ECAL_peaks,
                    REC_calorimeter, MC_RecMatch, REC_particle, MC_particle
                )

                # Write the data for each peak to the CSV file
                for mcindex, data in result.items():
                    # Determine the unique_mc_index
                    if mcindex == -1:
                        unique_mc_index = -1
                    else:
                        unique_mc_index = unique_mc_counter
                        unique_mc_counter += 1

                    mc_pid = data['mc_pid']

                    for peak in data['peaks']:
                        row = [
                            event_number, mcindex, unique_mc_index, mc_pid,
                            peak['id'], peak['status'], peak['sector'], peak['layer'],
                            peak['energy'], peak['time'], peak['xo'], peak['yo'], peak['zo'],
                            peak['xe'], peak['ye'], peak['ze'], peak['width'], peak['pindex'],
                            peak['rec_pid']
                        ]
                        csv_writer.writerow(row)

    def process_hipo(self):
        """
        Process the hipo file to generate a CSV with peak data and another CSV with intersection data.
        """
        # Step 1: Generate the initial peaks CSV
        self.read_ecal_data_from_event()
        
        print(f"Processing complete.")
