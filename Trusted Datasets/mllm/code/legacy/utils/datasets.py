import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

class ESMDataset(Dataset):
    def __init__(self, file_path=None, file_type='pth', partition='train', train_split=0.8, val_split=0.1, test_split=0.1, seq_len=240):
        """
        Initialize the dataset, loading from either a CSV or a .pth file, and partition it based on the specified dataset portion.
        """
        if file_type == 'csv': self.full_data = pd.read_csv(file_path, usecols=['aa_seq', 'deltaG'])
        elif file_type == 'pth': self.full_data = torch.load(file_path, usecols=['aa_seq', 'deltaG'])
        else: raise ValueError("file_type must be 'csv' or 'pth'")

        total_size = len(self.full_data)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        # Randomly split the dataset
        train_data, val_data, test_data = random_split(self.full_data, [train_size, val_size, test_size])

        if partition == 'train': self.data = train_data
        elif partition == 'val': self.data = val_data
        elif partition == 'test': self.data = test_data
        else: raise ValueError("partition must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract sequence and deltaG
        if isinstance(self.full_data, pd.DataFrame):
            # When full_data is a DataFrame, ensure it's accessed by iloc
            seq, deltaG = self.full_data.iloc[self.data.indices[idx]]
        else:
            # Assuming self.data is a subset with direct access to tensors or a list
            seq, deltaG = self.data[idx]
        return seq, deltaG

# Amino acid sequence encoding map
AA_IDX = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9,
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17,
    'W': 18, 'Y': 19, 'V': 20
}

class ESMnrgDataset(Dataset):
    def __init__(self, file_path=None, file_type='csv', partition='train', train_split=0.8, val_split=0.1, test_split=0.1, seq_len=240):
        """
        Initialize the dataset, loading from either a CSV or a .pth file, and partition it.

        Parameters:
            file_path (str): Path to the dataset file.
            file_type (str): Type of the file ('csv' or 'pth').
            partition (str): The part of the dataset to load ('train', 'val', or 'test').
            train_split (float): Fraction of the dataset to be used as training data.
            val_split (float): Fraction of the dataset to be used as validation data.
            test_split (float): Fraction of the dataset to be used as test data.
            seq_len (int): Target length for sequence padding.
        """
        if file_type == 'csv': self.full_data = pd.read_csv(file_path, usecols=['aa_seq', 'deltaG'])
        elif file_type == 'pth': self.full_data = torch.load(file_path, usecols=['aa_seq', 'deltaG'])
        else: raise ValueError("file_type must be 'csv' or 'pth'")

        self.seq_len = seq_len

        # Calculate sizes of each subset
        total_size = len(self.full_data)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size  # Ensure the entire dataset is covered

        # Randomly split the dataset
        train_data, val_data, test_data = random_split(self.full_data, [train_size, val_size, test_size])

        # Assign the appropriate dataset partition
        if partition == 'train':
            self.data = train_data
        elif partition == 'val':
            self.data = val_data
        elif partition == 'test':
            self.data = test_data
        else:
            raise ValueError("partition must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sequence and deltaG
        seq, deltaG = self.data[idx]

        # Convert amino acid sequence to indices
        seq_encoded = [AA_IDX.get(aa, 0) for aa in seq]  # 0 for unknown amino acids
        
        # Pad the sequence
        padded_seq = np.zeros(self.seq_len, dtype=int)
        padded_seq[:len(seq_encoded)] = seq_encoded[:self.seq_len]
        
        # Convert to tensors
        seq_tensor = torch.tensor(padded_seq, dtype=torch.long)
        deltaG_tensor = torch.tensor(deltaG, dtype=torch.float)
        
        return seq_tensor, deltaG_tensor
    
    def save_pth(self, pth_file, type='full'):
        if type == 'full':
            torch.save(self.full_data, pth_file)
        elif type == 'partition':
            torch.save(self.data, pth_file)
        else:
            raise ValueError("type must be 'full' or 'partition'")
        
    def save_csv(self, csv_file, type='full'):
        if type == 'full':
            self.full_data.to_csv(csv_file, index=False)
        elif type == 'partition':
            self.data.to_csv(csv_file, index=False)
        else:
            raise ValueError("type must be 'full' or 'partition'")