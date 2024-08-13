import numpy as np
import pandas as pd
import hipopy

class ECALDataReader:
    """
    Class to read and process ECAL data from a given file.

    Attributes:
    -----------
    file : hipopy.hipopy.open
        The file object to read data from.
    """

    def __init__(self, filename):
        """
        Initialize ECALDataReader with a file.
        
        Parameters:
        -----------
        filename : str
            The path to the file to be read.
        """
        self.file = hipopy.hipopy.open(filename, mode='r')

    def get_dict(self, bank):
        """
        Convert a bank of data from the file into a DataFrame.

        Parameters:
        -----------
        bank : str
            The name of the bank to retrieve data from.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the bank's data.
        """
        my_dict = {}
        branch_dict = self.file.getNamesAndTypes(bank)
        for branch, branchtype in branch_dict.items():
            my_dict[branch] = self.get_values(bank, branch, branchtype)
        return pd.DataFrame(my_dict)

    def get_values(self, bank, branch, branchtype):
        """
        Retrieve values from a specific branch of the bank.

        Parameters:
        -----------
        bank : str
            The name of the bank to retrieve data from.
        branch : str
            The specific branch in the bank.
        branchtype : str
            The type of the data in the branch (e.g., "B", "I", "F").

        Returns:
        --------
        np.array
            Array of values from the branch.
        """
        if branchtype == "B":
            values = np.array(self.file.getBytes(bank, branch))
        elif branchtype == "S":
            values = np.array(self.file.getShorts(bank, branch))
        elif branchtype == "L":
            values = np.array(self.file.getLongs(bank, branch))
        elif branchtype == "I":
            values = np.array(self.file.getInts(bank, branch))
        elif branchtype == "F":
            values = np.array(self.file.getFloats(bank, branch))
        else:
            raise ValueError("Unknown branchtype == ", branchtype)
        return values
