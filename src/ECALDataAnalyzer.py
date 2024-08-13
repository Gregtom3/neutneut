import pandas as pd
import numpy as np
from itertools import combinations
from ECALDataReader import ECALDataReader
from ECALDataProcessor import ECALDataProcessor
import csv
from tqdm import tqdm

class ECALDataAnalyzer:
    """
    Class to analyze ECAL data and write the results to a CSV file, including intersections.
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

    def calculate_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
        Calculate the intersection point between two lines.
        """
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel

        xi_num = (x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)
        yi_num = (x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)
        
        xi = xi_num / denom
        yi = yi_num / denom

        return xi, yi

    def analyze_ecal_peaks(self, df):
        """
        Analyze ECAL peaks from the DataFrame and find intersections between peak lines.
        """
        intersection_data = []

        # Group by event_number
        events = df.groupby('event_number')

        for event_number, event_df in events:
            # Split peaks into layer groups 1-3, 4-6, and 7-9
            for layer_group, group_df in event_df.groupby(pd.cut(event_df['peak_layer'], bins=[0, 3, 6, 9], labels=[1, 2, 3]), observed=False):
                hits = group_df[['peak_xo', 'peak_yo', 'peak_xe', 'peak_ye', 'peak_energy', 'peak_time', 'peak_layer', 'peak_sector', 'peak_pindex', 'peak_rec_pid', 'mc_index', 'mc_pid', 'unique_mc_index']].values
                
                for (xo1, yo1, xe1, ye1, energy1, time1, layer1, sector1, pindex1, rec_pid1, mc_index1, mc_pid1, unique_mc_index1), \
                    (xo2, yo2, xe2, ye2, energy2, time2, layer2, sector2, pindex2, rec_pid2, mc_index2, mc_pid2, unique_mc_index2) in combinations(hits, 2):
                    
                    # Calculate intersection
                    intersection = self.calculate_intersection(xo1, yo1, xe1, ye1, xo2, yo2, xe2, ye2)
                    
                    if intersection:
                        xi, yi = intersection
                        
                        # Check if intersection is within bounding box for both lines
                        if ((min(xo1, xe1) <= xi <= max(xo1, xe1) and min(yo1, ye1) <= yi <= max(yo1, ye1)) and
                            (min(xo2, xe2) <= xi <= max(xo2, xe2) and min(yo2, ye2) <= yi <= max(yo2, ye2))):

                            # Determine the final values for pindex, mc_index, mc_pid, and unique_mc_index
                            pindex = pindex1 if pindex1 == pindex2 else -1
                            mc_index = mc_index1 if mc_index1 == mc_index2 else -1
                            mc_pid = mc_pid1 if mc_pid1 == mc_pid2 else -1
                            unique_mc_index = unique_mc_index1 if unique_mc_index1 == unique_mc_index2 else -1

                            intersection_data.append({
                                'event_number': event_number,
                                'x': xi,
                                'y': yi,
                                'sector': sector1,
                                'layer_A': layer1,
                                'layer_B': layer2,
                                'energy_A': energy1,
                                'energy_B': energy2,
                                'time_A': time1,
                                'time_B': time2,
                                'pindex': pindex,
                                'mc_index': mc_index,
                                'mc_pid': mc_pid,
                                'unique_mc_index': unique_mc_index
                            })

        # Convert the intersection data into a DataFrame
        intersection_df = pd.DataFrame(intersection_data)
        return intersection_df

    def process_hipo(self):
        """
        Process the hipo file to generate a CSV with peak data and another CSV with intersection data.
        """
        # Step 1: Generate the initial peaks CSV
        self.read_ecal_data_from_event()

        # Step 2: Read the generated peaks CSV
        df = pd.read_csv(self.output_filename)

        # Step 3: Analyze peaks to find intersections and generate intersection CSV
        intersection_df = self.analyze_ecal_peaks(df)

        # Step 4: Save the intersection data to a new CSV file
        intersection_output_filename = self.output_filename.replace(".csv", "-intersections.csv")
        intersection_df.to_csv(intersection_output_filename, index=False)

        print(f"Processing complete. Intersection data saved to {intersection_output_filename}")
