# imports
from typing import List, Union, Optional
from torch.utils.data import DataLoader

# Custom imports
from code.utils.datasets import ESMnrgDataset, ESMDataset


def get_loader(file_path: str = "data/ds23_sm.csv", file_type: str = "csv", batch_size: int = 32, shuffle: bool = True, num_workers: int = 2, dataset: str = 'train', 
               train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1,
               verbose: int = 0, seq_len: int = 240) -> DataLoader:
    """
    Generate a DataLoader object for a specific dataset.

    Parameters:
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Flag to shuffle the dataset.
        num_workers (int): The number of workers for the DataLoader.
        dataset (str): The dataset to load ('train', 'val', or 'test').
        train_split (float): Proportion of data used for training.
        val_split (float): Proportion of data used for validation.
        test_split (float): Proportion of data used for testing.
        verbose (int): Level of verbosity in output logging.

    Returns:
        DataLoader: Configured DataLoader object for the specified dataset.
    """
    if verbose > 0:
        print(f"Setting up DataLoader for {dataset} set with batch size {batch_size}...")

    # Assuming ESMnrgDataset is appropriately defined elsewhere to handle splits based on dataset parameter
    dataset = ESMnrgDataset(file_path=file_path, file_type=file_type, partition=dataset, train_split=train_split, val_split=val_split, test_split=test_split, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader

def get_esm_loader(converter: callable, file_path: str = "data/ds23_sm.csv", file_type: str = "csv", batch_size: int = 32, shuffle: bool = True, num_workers: int = 2, dataset: str = 'train', 
               train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1,
               verbose: int = 0, seq_len: int = 240) -> DataLoader:
    """
    Generate a DataLoader object for a specific dataset, utilizing ESM-2's batch converter for processing.
    """
    if verbose > 0:
        print(f"Setting up DataLoader for {dataset} set with batch size {batch_size}...")

    # Load the dataset with appropriate parameters
    dataset_instance = ESMDataset(file_path=file_path, file_type=file_type, partition=dataset, train_split=train_split, val_split=val_split, test_split=test_split, seq_len=seq_len)
    
    # Custom collate function using ESM-2's batch converter
    def collate_fn(batch):
        # batch_labels, batch_strs, batch_tokens = converter([(f"deltaG:{x[1]}", x[0]) for i, x in enumerate(batch)])
        batch_labels, batch_strs, batch_tokens = converter([(f"deltaG:{x[1]}", x[0]) for i, x in enumerate(batch)])
        return batch_labels, batch_strs, batch_tokens
    
    loader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return loader

