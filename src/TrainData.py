import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from itertools import combinations
import h5py
import os

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
            
        # Group by file_number and file_event_number
        grouped = df.groupby(['file_number', 'file_event_number'])

        # Filter out groups where the number of non- -1 unique otid values is less than min_particles
        def filter_group(group):
            non_negative_unique_otid_count = group[group['otid'] != -1]['otid'].nunique()
            return non_negative_unique_otid_count >= min_particles

        df = grouped.filter(filter_group)

        df = self._filter_peak_time(df)
        df = self._one_hot_encode(df, 'sector', 6)
        df = self._one_hot_encode(df, 'layer', 9)
        df = self._rescale_columns(df, ['energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze', 'centroid_x', 'centroid_y'])
        df = self._delete_columns(df, ['otid', 'event_number', 'status', 'id'])
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
        return df[(df['time'] >= 0) & (df['time'] <= 200) & (df['time'] >= 0) & (df['time'] >= 0)].copy()

    def _rescale_columns(self, df, columns_to_scale):
        """
        Rescale specified numeric columns between 0-1, with certain groups sharing the same min-max scaling.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to rescale.
        columns_to_scale : list of str
            List of column names to scale.
        Returns:
        --------
        pd.DataFrame
            The DataFrame with scaled columns.
        """
        # Group 1: Shared scaling for 'xo', 'yo', 'xe', 'ye', 'centroid_x' and 'centroid_y'
        group_1 = ['xo', 'yo', 'xe', 'ye', 'centroid_x', 'centroid_y']
        group_1_values = df[group_1].values.flatten()  # Flatten to find global min and max
        min_val = group_1_values.min()
        max_val = group_1_values.max()
        df[group_1] = (df[group_1] - min_val) / (max_val - min_val)

        # Group 2: Shared scaling for 'zo' and 'ze'
        group_2 = ['zo', 'ze']
        group_2_values = df[group_2].values.flatten()  # Flatten to find global min and max
        min_val = group_2_values.min()
        max_val = group_2_values.max()
        df[group_2] = (df[group_2] - min_val) / (max_val - min_val)

        # Other columns to scale individually
        other_columns = [col for col in columns_to_scale if col not in group_1 + group_2]
        
        df[other_columns] = self.scaler.fit_transform(df[other_columns])

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
            'file_number', 'file_event_number', 'unique_otid', 'mc_pid'
        ] + [col for col in df.columns if col not in [
            'file_number', 'file_event_number', 'unique_otid', 'mc_pid'
        ]]

        # Reorder the columns
        df = df[column_order]
        
        return df
    
    
    
class TrainData:
    """
    Class to handle and merge multiple intersection CSV files into a single DataFrame,
    ensuring unique mc_index across all files and splitting the data into train and test sets.
    """

    def __init__(self, csv_files, train_size=0.8, return_tensor=False, K=10, remove_background=False, min_particles=1):
        """
        Initialize the TrainData class with a list of CSV files and the desired train/test split.

        Parameters:
        -----------
        csv_files : list[str]
            List of paths to the intersection CSV files.
        train_size : float
            The proportion of the dataset to include in the train split (default is 0.8).
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
        self.train_size = train_size
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
        self._split_data()


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
            df['file_event_number'] = df['event_number']

            # Create the unique_otid column
            df['unique_otid'] = df['otid']

            # Adjust the otid values for uniqueness across files, skipping -1
            non_negative_mask = df['unique_otid'] != -1

            # Combine file_event_number, otid, and file_number to ensure uniqueness
            df.loc[non_negative_mask, 'unique_otid'] = (
                df.loc[non_negative_mask, 'file_event_number'].astype(str) + "_" +
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

    def _split_data(self):
        event_groups = self.data.groupby(['file_number', 'file_event_number'])
        # Get a list of unique events
        unique_events = event_groups.size().index.tolist()

        # Shuffle the events
        np.random.seed(42)
        np.random.shuffle(unique_events)

        # Handle the case where 100% of data is to be used as training data
        if self.train_size >= 1.0:
            train_events = unique_events
            test_events = []
        else:
            # Calculate the number of events for the training set
            train_event_count = int(len(unique_events) * self.train_size)
            # Split the events into train and test sets
            train_events = unique_events[:train_event_count]
            test_events = unique_events[train_event_count:]

        # Select data corresponding to the train and test events
        self.train_data = self.data[self.data.set_index(['file_number', 'file_event_number']).index.isin(train_events)]
        self.test_data = self.data[self.data.set_index(['file_number', 'file_event_number']).index.isin(test_events)]

        if self.return_tensor:
            self.train_data = self._convert_to_tensor(self.train_data)
            self.test_data = self._convert_to_tensor(self.test_data) if len(self.test_data) > 0 else None

    def _convert_to_tensor(self, df):
        feature_columns = [
            'energy', 'time', 'xo', 'yo', 'zo', 'xe', 'ye', 'ze',
            'layer_1', 'layer_2', 'layer_3', 'layer_4',
            'layer_5', 'layer_6', 'layer_7', 'layer_8', 'layer_9', 
            'centroid_x', 'centroid_y', 'is_3way_same_group', 'is_2way_same_group', 'is_3way_cross_group', 'is_2way_cross_group',
            'sector_1', 'sector_2', 'sector_3', 
            'sector_4','sector_5', 'sector_6', 
            'rec_pid', 'pindex', 'mc_pid','unique_otid'
        ]

        tensors = []
        grouped = df.groupby(['file_number', 'file_event_number'])
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

    def get_train_data(self, maxN=-1):
        return self._get_data(self.train_data, maxN)

    def get_test_data(self, maxN=-1):
        return self._get_data(self.test_data, maxN)

    def _get_data(self, data, maxN):
        """
        Retrieve processed data, potentially limited by maxN.

        Parameters:
        -----------
        data : np.array
            The data array to process.
        maxN : int
            Maximum number of samples to return. If maxN <= 0, return all data.

        Returns:
        --------
        tuple of np.array
            Returns X, y, and misc data arrays.
        """
        if maxN > 0:
            data = data[:maxN]

        X = data[:, :, :29]
        y = data[:, :, -1:]
        misc = data[:, :, 29:-1]

        return X, y, misc

    
    
    
    
def load_train_test_data(directory):
    # Define filenames
    train_filename = os.path.join(directory, 'dataset_train.h5')
    test_filename = os.path.join(directory, 'dataset_test.h5')

    # Load testing data
    with h5py.File(test_filename, 'r') as hf:
        test_X = hf['test_X'][:]
        test_y = hf['test_y'][:]
        test_misc = hf['test_misc'][:]
    print(f"Testing data loaded from {test_filename}")

    # Load training data
    with h5py.File(train_filename, 'r') as hf:
        train_X = hf['train_X'][:]
        train_y = hf['train_y'][:]
        train_misc = hf['train_misc'][:]
    print(f"Training data loaded from {train_filename}")

    return (train_X, train_y, train_misc), (test_X, test_y, test_misc)