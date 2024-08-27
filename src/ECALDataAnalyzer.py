import pandas as pd
import numpy as np
from itertools import combinations
from ECALDataReader import ECALDataReader
from ECALDataProcessor import ECALDataProcessor
import csv
from tqdm import tqdm


class ECALDataAnalyzer:
    """
    Class to analyze ECAL data and write the results to a CSV file.
    """

    def __init__(self, input_filename, output_filename="output.csv"):
        """
        Initialize the analyzer with input and output file names.
        
        Parameters:
        -----------
        input_filename : str
            Path to the input hipo file.
        output_filename : str
            Path to the initial output CSV file for hits (default is "output.csv").
        """
        self.input_filename = input_filename
        self.output_filename = output_filename

    def read_ecal_data_from_event(self):
        """
        Read and process ECAL data from an event file and write results to a CSV.
        """
        reader = ECALDataReader(self.input_filename)
        
        # Open the CSV file for writing
        with open(self.output_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            header = [
                "event_number", "id", "mc_pid", "otid", "sector", "layer", "energy", "time", 
                "xo", "yo", "zo", "xe", "ye", "ze", "rec_pid", "pindex"
            ]
            csv_writer.writerow(header)

            # Loop over the events and process the data
            for event_number, event in tqdm(enumerate(reader.file)):
                ECAL_hits = reader.get_dict("ECAL::hits+")
                if(len(ECAL_hits)==0): # Skip events without ECAL hits
                    continue
                REC_cal = reader.get_dict("REC::Calorimeter")
                REC_parts = reader.get_dict("REC::Particle")

                # Apply the rec_pid and pindex determination
                def get_rec_pid_and_pindex(row):
                    if row['clusterId'] == -1:
                        return pd.Series({'rec_pid': -1, 'pindex': -1})
                    else:
                        pindex = REC_cal[REC_cal["index"] == row['clusterId'] - 1]["pindex"].values[0]
                        rec_pid = REC_parts["pid"].values[pindex]
                        return pd.Series({'rec_pid': rec_pid, 'pindex': pindex})

                # Add rec_pid and pindex columns to ECAL_hits DataFrame
                ECAL_hits[['rec_pid', 'pindex']] = ECAL_hits.apply(get_rec_pid_and_pindex, axis=1)

                # Write the data for each hit to the CSV file
                for _, hit in ECAL_hits.iterrows():
                    row = [
                        event_number, hit['id'], hit['pid'], hit['otid'], hit['sector'], hit['layer'], hit['energy'], hit['time'],
                        hit['xo'], hit['yo'], hit['zo'], hit['xe'], hit['ye'], hit['ze'], hit['rec_pid'], hit['pindex']
                    ]
                    csv_writer.writerow(row)

    def process_hipo(self):
        """
        Process the hipo file to generate a CSV with hits data.
        """
        # Generate the initial hits CSV
        self.read_ecal_data_from_event()
        
        # Add the centroid column
        self.add_centroid_columns(R=20)
        self.add_cluster_centroid_columns()
        # Generate intersections CSV
        self.generate_intersections_csv(n_values=[1, 4, 7], R=20)
        print(f"Processing complete.")
        
    def generate_intersections_csv(self, n_values=[1, 4, 7], R=0.5):
        """
        Generate an intersections CSV based on the previously created CSV file.
        
        Parameters:
        -----------
        n_values : list of int
            The layers to consider for intersection calculations.
        R : float
            The radius within which the centroid must lie for the intersection to be recorded.
        """
        # Load the hits data from the previously generated CSV
        df = pd.read_csv(self.output_filename)
        
        # Calculate intersections
        intersection_df = calculate_centroid_and_intersection(df, n_values, R)
        
        # Define the new filename for the intersections CSV
        intersections_filename = self.output_filename.replace(".csv", "-intersections.csv")
        
        # Save the intersections DataFrame to the new CSV
        intersection_df.to_csv(intersections_filename, index=False)
        
        print(f"Intersections CSV generated: {intersections_filename}")
        
    def add_centroid_columns(self, R=0.5):
        """
        Add centroid_x and centroid_y columns to the original CSV file based on the intersection 
        of strips across layers within the specified layer groups.

        Parameters:
        -----------
        R : float
            The radius within which the centroid must lie for the intersection to be valid.
        """
        # Load the hits data from the previously generated CSV
        df = pd.read_csv(self.output_filename)

        # Initialize centroid columns with default values
        df['centroid_x'] = 0.0
        df['centroid_y'] = 0.0

        # Group the data by event number
        grouped = df.groupby(['event_number','sector'])

        # Iterate through each event
        for (event_number,sector), group in grouped:
            for n_start in [1, 4, 7]:
                # Select the relevant layers
                layer_n = group[group['layer'] == n_start]
                layer_n1 = group[group['layer'] == n_start + 1]
                layer_n2 = group[group['layer'] == n_start + 2]

                for _, strip_n in layer_n.iterrows():
                    max_energy = 0
                    best_centroid = np.array([0.0, 0.0])

                    # Try to find 3-way intersections first
                    for _, strip_n1 in layer_n1.iterrows():
                        for _, strip_n2 in layer_n2.iterrows():
                            closest_n_n1 = closest_point_between_lines(
                                np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n['xe'], strip_n['ye']]) - np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n1['xo'], strip_n1['yo']]),
                                np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']])
                            )
                            closest_n_n2 = closest_point_between_lines(
                                np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n['xe'], strip_n['ye']]) - np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n2['xo'], strip_n2['yo']]),
                                np.array([strip_n2['xe'], strip_n2['ye']]) - np.array([strip_n2['xo'], strip_n2['yo']])
                            )
                            closest_n1_n2 = closest_point_between_lines(
                                np.array([strip_n1['xo'], strip_n1['yo']]),
                                np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']]),
                                np.array([strip_n2['xo'], strip_n2['yo']]),
                                np.array([strip_n2['xe'], strip_n2['ye']]) - np.array([strip_n2['xo'], strip_n2['yo']])
                            )

                            centroid = (closest_n_n1 + closest_n_n2 + closest_n1_n2) / 3
                            distances = [
                                np.linalg.norm(closest_n_n1 - centroid),
                                np.linalg.norm(closest_n_n2 - centroid),
                                np.linalg.norm(closest_n1_n2 - centroid)
                            ]

                            if all(distance <= R for distance in distances):
                                total_energy = strip_n['energy'] + strip_n1['energy'] + strip_n2['energy']
                                if total_energy > max_energy:
                                    max_energy = total_energy
                                    best_centroid = centroid

                    # If no valid 3-way intersection found, try to find 2-way intersections
                    if max_energy == 0:
                        for _, strip_n1 in layer_n1.iterrows():
                            closest_n_n1 = closest_point_between_lines(
                                np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n['xe'], strip_n['ye']]) - np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n1['xo'], strip_n1['yo']]),
                                np.array([strip_n1['xe'], strip_n1['ye']]) - np.array([strip_n1['xo'], strip_n1['yo']])
                            )
                            centroid = (closest_n_n1 + np.array([strip_n['xo'], strip_n['yo']]) + np.array([strip_n1['xo'], strip_n1['yo']])) / 3
                            distance = np.linalg.norm(closest_n_n1 - centroid)

                            if distance <= R:
                                total_energy = strip_n['energy'] + strip_n1['energy']
                                if total_energy > max_energy:
                                    max_energy = total_energy
                                    best_centroid = centroid

                        for _, strip_n2 in layer_n2.iterrows():
                            closest_n_n2 = closest_point_between_lines(
                                np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n['xe'], strip_n['ye']]) - np.array([strip_n['xo'], strip_n['yo']]),
                                np.array([strip_n2['xo'], strip_n2['yo']]),
                                np.array([strip_n2['xe'], strip_n2['ye']]) - np.array([strip_n2['xo'], strip_n2['yo']])
                            )
                            centroid = (closest_n_n2 + np.array([strip_n['xo'], strip_n['yo']]) + np.array([strip_n2['xo'], strip_n2['yo']])) / 3
                            distance = np.linalg.norm(closest_n_n2 - centroid)

                            if distance <= R:
                                total_energy = strip_n['energy'] + strip_n2['energy']
                                if total_energy > max_energy:
                                    max_energy = total_energy
                                    best_centroid = centroid

                    # Update the centroid columns in the DataFrame
                    idx = df[(df['event_number'] == event_number) & (df['id'] == strip_n['id'])].index
                    df.loc[idx, 'centroid_x'] = best_centroid[0]
                    df.loc[idx, 'centroid_y'] = best_centroid[1]

        # Save the updated DataFrame back to the original CSV file
        df.to_csv(self.output_filename, index=False)
        print(f"Centroid columns added and CSV saved: {self.output_filename}")
        

        
    def add_cluster_centroid_columns(self):
        """
        Add cluster_centroid_x and cluster_centroid_y columns to the original CSV file based on 
        grouping by event_number and otid, and calculating the mean of centroid_x and centroid_y 
        for the dominant sector.
        """
        # Load the hits data from the previously generated CSV
        df = pd.read_csv(self.output_filename)

        # Initialize cluster_centroid columns with default values
        df['cluster_centroid_x'] = 0.0
        df['cluster_centroid_y'] = 0.0

        # Group the data by event_number and otid
        grouped = df.groupby(['event_number', 'otid'])

        # Iterate through each group
        for (event_number, otid), group in grouped:
            if group.empty:
                continue
            
            # Determine the most common sector
            sector_counts = group['sector'].value_counts()
            dominant_sector = sector_counts.idxmax()

            # Check if the dominant sector has more than 40% of the hits
            if sector_counts[dominant_sector] > len(group) / 40:
                dominant_group = group[group['sector'] == dominant_sector]
            else:
                dominant_group = group  # No dominant sector, use all hits

            # Calculate the mean of non-zero centroid_x and centroid_y for the dominant group
            non_zero_centroids = dominant_group[(dominant_group['centroid_x'] != 0) & (dominant_group['centroid_y'] != 0)]
            if not non_zero_centroids.empty:
                mean_centroid_x = non_zero_centroids['centroid_x'].mean()
                mean_centroid_y = non_zero_centroids['centroid_y'].mean()
            else:
                mean_centroid_x = 0.0
                mean_centroid_y = 0.0

            # Assign the mean centroids to all rows in the group
            df.loc[(df['event_number'] == event_number) & (df['otid'] == otid), 'cluster_centroid_x'] = mean_centroid_x
            df.loc[(df['event_number'] == event_number) & (df['otid'] == otid), 'cluster_centroid_y'] = mean_centroid_y

        # Save the updated DataFrame back to the original CSV file
        df.to_csv(self.output_filename, index=False)
        print(f"Cluster centroid columns added and CSV saved: {self.output_filename}")
        
        
def closest_point_between_lines(a1, b1, a2, b2):
    """
    Calculate the closest point between two lines in 2D space.
    
    Parameters:
    -----------
    a1, b1 : numpy.ndarray
        The starting point and direction vector for the first line.
    a2, b2 : numpy.ndarray
        The starting point and direction vector for the second line.
        
    Returns:
    --------
    numpy.ndarray
        The closest point between the two lines.
    """
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    a_diff = a2 - a1
    det = b1[0] * b2[1] - b1[1] * b2[0]
    
    if np.isclose(det, 0):
        lambda_1 = np.dot(a_diff, b1)
        closest_point_1 = a1 + lambda_1 * b1
        return closest_point_1
    else:
        lambda_ = (a_diff[0] * b2[1] - a_diff[1] * b2[0]) / det
        closest_point = a1 + lambda_ * b1
        return closest_point

def calculate_centroid_and_intersection(df, n_values, R=0.5):
    """
    Calculate the centroid of closest points between lines in different layers
    and determine if they form valid intersections.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing line data with columns ['xo', 'yo', 'xe', 'ye', 'layer', etc.].
    n_values : list of int
        List of layer indices to consider for intersections.
    R : float
        Radius within which the centroid must lie for the intersection to be valid.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing valid intersections with centroid positions and associated data.
    """
    results = []

    grouped = df.groupby('event_number')

    for event, group in grouped:
        for n in n_values:
            layer_n = group[group['layer'] == n]
            layer_n1 = group[group['layer'] == n+1]
            layer_n2 = group[group['layer'] == n+2]

            for _, lineA in layer_n.iterrows():
                for _, lineB in layer_n1.iterrows():
                    for _, lineC in layer_n2.iterrows():
                        # Calculate the closest points
                        aA, bA = np.array([lineA['xo'], lineA['yo']]), np.array([lineA['xe'], lineA['ye']]) - np.array([lineA['xo'], lineA['yo']])
                        aB, bB = np.array([lineB['xo'], lineB['yo']]), np.array([lineB['xe'], lineB['ye']]) - np.array([lineB['xo'], lineB['yo']])
                        aC, bC = np.array([lineC['xo'], lineC['yo']]), np.array([lineC['xe'], lineC['ye']]) - np.array([lineC['xo'], lineC['yo']])
                        
                        closest_AB = closest_point_between_lines(aA, bA, aB, bB)
                        closest_AC = closest_point_between_lines(aA, bA, aC, bC)
                        closest_BC = closest_point_between_lines(aB, bB, aC, bC)
                        
                        centroid = (closest_AB + closest_AC + closest_BC) / 3
                        
                        distances = [
                            np.linalg.norm(closest_AB - centroid),
                            np.linalg.norm(closest_AC - centroid),
                            np.linalg.norm(closest_BC - centroid)
                        ]
                        
                        if all(distance <= R for distance in distances):
                            mc_pid = lineA['mc_pid'] if lineA['mc_pid'] == lineB['mc_pid'] == lineC['mc_pid'] else -1
                            otid = lineA['otid'] if lineA['otid'] == lineB['otid'] == lineC['otid'] else -1
                            sector = lineA['sector'] if lineA['sector'] == lineB['sector'] == lineC['sector'] else -1
                            rec_pid = lineA['rec_pid'] if lineA['rec_pid'] == lineB['rec_pid'] == lineC['rec_pid'] else -1
                            pindex = lineA['pindex'] if lineA['pindex'] == lineB['pindex'] == lineC['pindex'] else -1
                            if sector == -1:  # The cluster must come from intersecting strips in the same sector
                                continue
                            results.append({
                                'event_number': event,
                                'centroid_x': centroid[0],
                                'centroid_y': centroid[1],
                                'time_A'  : lineA['time'],
                                'time_B'  : lineB['time'],
                                'time_C'  : lineC['time'],
                                'energy_A'  : lineA['energy'],
                                'energy_B'  : lineB['energy'],
                                'energy_C'  : lineC['energy'],
                                'layer': 1+(n-1)/3,
                                'sector': sector,
                                'mc_pid': mc_pid,
                                'otid': otid,
                                'rec_pid': rec_pid,
                                'pindex': pindex,
                                'xo_A': lineA['xo'], 'yo_A': lineA['yo'], 'xe_A': lineA['xe'], 'ye_A': lineA['ye'],
                                'xo_B': lineB['xo'], 'yo_B': lineB['yo'], 'xe_B': lineB['xe'], 'ye_B': lineB['ye'],
                                'xo_C': lineC['xo'], 'yo_C': lineC['yo'], 'xe_C': lineC['xe'], 'ye_C': lineC['ye'],
                            })
    return pd.DataFrame(results)