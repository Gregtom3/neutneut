import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from itertools import combinations
import h5py
import os
from global_params import *
import random

class DataPreprocessor:
    """
    Class to preprocess the data, including filtering, rescaling, column removal, rotation, and one-hot encoding operations.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, df, remove_background=True, min_particles=1):
        """
        Preprocess the DataFrame by calling individual preprocessing subroutines.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to preprocess.
        remove_background : bool
            If True, polish dataset to remove NaNs, backgrounds, etc.
        min_particles : int
            Minimum number of particles per event
        Returns:
        --------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        if remove_background:
            # Remove rows where 'otid' is -1 and remove rows where 'centroid_x' or 'centroid_y' is 0
            df = df[df['otid'] != -1]
            df = df[(df['centroid_x'] != 0) & (df['centroid_y'] != 0)]
            
            # Remove rows containing any NaN values
            df = df.dropna()
            
        # Group by file_number and file_event
        grouped = df.groupby(['file_number', 'file_event'])

        # Filter out groups where the number of non- -1 unique otid values is less than min_particles
        def filter_group(group):
            non_negative_unique_otid_count = group[group['otid'] != -1]['otid'].nunique()
            return non_negative_unique_otid_count >= min_particles

        df = grouped.filter(filter_group)

        df = self._filter_peak_time(df)
        df = self._one_hot_encode(df, 'sector', 6)
        df = self._one_hot_encode(df, 'layer', 9)
        df = self._rescale_columns(df)
        df = self._delete_columns(df, ['otid', 'event', 'status', 'id'])
        df = self._reorder_columns(df)

        return df
    
    def _filter_peak_time(self, df):
        """
        Filter rows where peak time is outside the range 0-200.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to filter.
        Returns:
        --------
        pd.DataFrame
            The filtered DataFrame.
        """
        return df[(df['time'] >= ECAL_time_min) & (df['time'] <= ECAL_time_max)].copy()

    def _rescale_columns(self, df):
        """
        Rescale specified numeric columns, with certain groups sharing the same scaling range.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to rescale.
            
        Returns:
        --------
        pd.DataFrame
            The DataFrame with scaled columns.
        """

        # Group 1: Shared scaling for 'xo', 'yo', 'xe', 'ye', 'centroid_x', 'centroid_y'
        group_1 = ['xo', 'yo', 'xe', 'ye', 'centroid_x', 'centroid_y']
        min_val_1, max_val_1 = ECAL_xy_min, ECAL_xy_max  # Define the fixed range for group 1
        df[group_1] = (df[group_1] - min_val_1) / (max_val_1 - min_val_1)

        # Group 2: Shared scaling for 'zo', 'ze', 'centroid_z'
        group_2 = ['zo', 'ze', 'centroid_z']
        min_val_2, max_val_2 = ECAL_z_min, ECAL_z_max  # Define the fixed range for group 2
        df[group_2] = (df[group_2] - min_val_2) / (max_val_2 - min_val_2)

        # Group 3: Shared scaling for energy columns
        group_3 = ['energy']
        min_val_3, max_val_3 = ECAL_energy_min, ECAL_energy_max  # Define the fixed range for group 3
        df[group_3] = (df[group_3] - min_val_3) / (max_val_3 - min_val_3)

        # Group 4: Shared scaling for time columns
        group_4 = ['time'] 
        min_val_4, max_val_4 = ECAL_time_min, ECAL_time_max  # Define the fixed range for group 4
        df[group_4] = (df[group_4] - min_val_4) / (max_val_4 - min_val_4)

        return df

    def _delete_columns(self, df, columns_to_delete):
        """
        Delete specified columns from the DataFrame if they exist.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to modify.
        columns_to_delete : list of str
            List of column names to delete if they exist in the DataFrame.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with specified columns removed.
        """
        return df.drop(columns=[col for col in columns_to_delete if col in df.columns])
    
    
    def _one_hot_encode(self, df, column_name, num_categories):
        """
        One-hot encode a specified column.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to modify.
        column_name : str
            The name of the column to one-hot encode.
        num_categories : int
            The number of unique categories in the column.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with one-hot encoded columns.
        """
        for i in range(1, num_categories + 1):
            df[f'{column_name}_{i}'] = (df[column_name] == i).astype(int)
        df = df.drop(columns=[column_name])
        return df    


    def _reorder_columns(self, df):
        """
        Reorder the DataFrame columns

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to reorder.

        Returns:
        --------
        pd.DataFrame
            The reordered DataFrame.
        """
        # Define the new order of columns
        column_order = [
            'file_number', 'file_event', 'unique_otid', 'mc_pid'
        ] + [col for col in df.columns if col not in [
            'file_number', 'file_event', 'unique_otid', 'mc_pid'
        ]]

        # Reorder the columns
        df = df[column_order]
        
        return df
    
    
    
class TrainData:
    """
    Class to handle and merge multiple intersection CSV files into a single DataFrame,
    ensuring unique mc_index across all files and splitting the data into train and test sets.
    """

    def __init__(self, csv_files, return_tensor=False, K=10, remove_background=False, min_particles=1):
        """
        Initialize the TrainData class with a list of CSV files

        Parameters:
        -----------
        csv_files : list[str]
            List of paths to the intersection CSV files.
        return_tensor : bool
            If True, returns the data as tensors instead of DataFrames.
        K : int
            The number of elements to include in each tensor along the second dimension.
        remove_background : bool
            If True, remove rows where the mc_index==-1 (peaks not associated with an MC::Particle)
        min_particles : int
            Minimum number of particles per event 
        """
        self.csv_files = csv_files
        self.return_tensor = return_tensor
        self.K = K
        self.remove_background = remove_background
        self.min_particles    = min_particles
        self.data = pd.DataFrame()
        self.train_data = None
        self.test_data = None

        # Automatically load, merge, preprocess, and split the data upon initialization
        self._load_and_merge_csvs()
        self._preprocess_data()


    def _load_and_merge_csvs(self):
        unique_otid_offset = 0
        merged_data = []

        for file_number, csv_file in tqdm(enumerate(self.csv_files), total=len(self.csv_files)):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

            # Skip empty files
            if len(df) == 0:
                continue

            df['file_number'] = file_number
            df['file_event'] = df['event']

            # Create the unique_otid column
            df['unique_otid'] = df['otid']

            # Adjust the otid values for uniqueness across files, skipping -1
            non_negative_mask = df['unique_otid'] != -1

            # Combine file_event, otid, and file_number to ensure uniqueness
            df.loc[non_negative_mask, 'unique_otid'] = (
                df.loc[non_negative_mask, 'file_event'].astype(str) + "_" +
                df.loc[non_negative_mask, 'otid'].astype(str) + "_" +
                df.loc[non_negative_mask, 'file_number'].astype(str)
            ).astype('category').cat.codes + unique_otid_offset

            # Update the offset for the next file, skipping the -1 value
            if df.loc[non_negative_mask, 'unique_otid'].max() != -1:
                unique_otid_offset = df.loc[non_negative_mask, 'unique_otid'].max() + 1

            merged_data.append(df)

        self.data = pd.concat(merged_data, ignore_index=True)

        if self.data.empty:
            raise ValueError("No data available after loading and merging CSV files. Please check your input files or filtering criteria.")

        print(f"Total files processed: {len(self.csv_files)}")

    def _preprocess_data(self):
        preprocessor = DataPreprocessor()
        self.data = preprocessor.preprocess(self.data, self.remove_background, self.min_particles)
        self.data = self._convert_to_tensor(self.data)

    def _convert_to_tensor(self, df):
        feature_columns = [
            'energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
            'layer_1', 'layer_2', 'layer_3', 'layer_4',
            'layer_5', 'layer_6', 'layer_7', 'layer_8', 'layer_9', 
            'centroid_x', 'centroid_y', 'centroid_z', 'is_3way_same_group', 'is_2way_same_group', 'is_3way_cross_group', 'is_2way_cross_group',
            'sector_1', 'sector_2', 'sector_3', 
            'sector_4','sector_5', 'sector_6', 
            'rec_pid', 'pindex', 'mc_pid','unique_otid'
        ]

        tensors = []
        grouped = df.groupby(['file_number', 'file_event'])
        for _, group in tqdm(grouped):
            group = group.sort_values(by='energy', ascending=False)
            
            tensor = group[feature_columns].values[:self.K]

            if len(tensor) < self.K:
                # Create padding with zeros for all columns except the last one
                padding = np.zeros((self.K - len(tensor), len(feature_columns)))
                padding[:, -1] = -1  # Set the last column to -1 in the padding

                tensor = np.vstack([tensor, padding])

            tensors.append(tensor)

        return tf.convert_to_tensor(tensors, dtype=tf.float32)

    def get_data(self, maxN=-1):
        if maxN > 0:
            data = self.data[:maxN]
        else:
            data = self.data
        
        X = data[:, :, :30]
        y = data[:, :, -1:]
        misc = data[:, :, 30:-1]

        return X, y, misc

def load_unzip_data(h5_filename):
    with h5py.File(h5_filename,'r') as hf:
        X = hf["X"][:]
        y = hf["y"][:]
        misc = hf["misc"][:]
    return (X,y,misc)


def load_zip_train_test_data(directory, batch_size, num_train_batches=None, num_test_batches=None):
    # List all .h5 files in the directory
    h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]

    # Check if there is only a single file
    single_file_split = len(h5_files) == 1

    # Shuffle files and split into 80/20 train-test split
    if not single_file_split:
        random.shuffle(h5_files)
        split_idx = int(train_test_ratio * len(h5_files))
        train_files = h5_files[:split_idx]
        test_files = h5_files[split_idx:]
    else:
        # When there is only one file, we'll split it internally
        train_files = h5_files
        test_files = []  # No separate test files, splitting happens within the file

    # Helper function to load each component (X, y, misc) in batches from multiple files or single file
    def load_data_component_in_batches(file_list, batch_size, dataset_name, num_batches=None, single_file_split=False):
        def generator():
            batch_count = 0
            for h5_filename in file_list:
                with h5py.File(h5_filename, 'r') as hf:
                    dataset = hf[dataset_name]
                    num_samples = len(dataset)

                    if single_file_split:
                        # Perform an 80/20 split within the single file
                        split_idx = int(0.8 * num_samples)
                        for i in range(0, split_idx, batch_size):
                            if num_batches and batch_count >= num_batches:
                                return
                            yield dataset[i:i+batch_size]
                            batch_count += 1
                    else:
                        # Yield all data from multiple files
                        for i in range(0, num_samples, batch_size):
                            if num_batches and batch_count >= num_batches:
                                return
                            yield dataset[i:i+batch_size]
                            batch_count += 1
        
        # Determine the shape based on the dataset_name (adjust shape based on actual data)
        if "X" in dataset_name:
            shape = (None, 100, 30)  # Example shape for 'X' data
        elif "y" in dataset_name:
            shape = (None, 100, 1)   # Example shape for 'y' data
        else:
            shape = (None, 100, 3)   # Example shape for 'misc' data
        
        # Create a TensorFlow Dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32)
        )
        return dataset

    # Helper function to load the test set from within a single file (for single file case)
    def load_test_data_component_from_single_file(file_list, batch_size, dataset_name, num_batches=None):
        def generator():
            batch_count = 0
            for h5_filename in file_list:
                with h5py.File(h5_filename, 'r') as hf:
                    dataset = hf[dataset_name]
                    num_samples = len(dataset)
                    split_idx = int(0.8 * num_samples)

                    for i in range(split_idx, num_samples, batch_size):
                        if num_batches and batch_count >= num_batches:
                            return
                        yield dataset[i:i+batch_size]
                        batch_count += 1

        # Determine the shape based on the dataset_name
        if "X" in dataset_name:
            shape = (None, 100, 30)
        elif "y" in dataset_name:
            shape = (None, 100, 1)
        else:
            shape = (None, 100, 3)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=shape, dtype=tf.float32)
        )
        return dataset

    # Load training components with optional limit on number of batches
    train_X_data = load_data_component_in_batches(train_files, batch_size, 'X', num_train_batches, single_file_split)
    train_y_data = load_data_component_in_batches(train_files, batch_size, 'y', num_train_batches, single_file_split)
    train_misc_data = load_data_component_in_batches(train_files, batch_size, 'misc', num_train_batches, single_file_split)

    # Load testing components: Either from separate test files or from within a single file
    if single_file_split:
        test_X_data = load_test_data_component_from_single_file(train_files, batch_size, 'X', num_test_batches)
        test_y_data = load_test_data_component_from_single_file(train_files, batch_size, 'y', num_test_batches)
        test_misc_data = load_test_data_component_from_single_file(train_files, batch_size, 'misc', num_test_batches)
    else:
        test_X_data = load_data_component_in_batches(test_files, batch_size, 'X', num_test_batches)
        test_y_data = load_data_component_in_batches(test_files, batch_size, 'y', num_test_batches)
        test_misc_data = load_data_component_in_batches(test_files, batch_size, 'misc', num_test_batches)

    # Return the datasets along with the sizes
    return (train_X_data, train_y_data, train_misc_data), (test_X_data, test_y_data, test_misc_data)

