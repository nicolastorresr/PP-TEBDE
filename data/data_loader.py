import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import PPTEBDEDataset

def load_dataset(dataset_name, config):
    """
    Load and preprocess the specified dataset.
    """
    if dataset_name == 'edurec':
        return load_edurec_dataset(config)
    elif dataset_name == 'movielens':
        return load_movielens_dataset(config)
    elif dataset_name == 'amazon':
        return load_amazon_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_edurec_dataset(config):
    """
    Load and preprocess the EduRec dataset.
    """
    # Load raw data
    df = pd.read_csv(config.edurec_path)
    
    # Preprocess
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to Unix timestamp
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['item_id'] = df['item_id'].astype('category').cat.codes
    
    return create_train_val_test_split(df, config)

def load_movielens_dataset(config):
    """
    Load and preprocess the MovieLens dataset.
    """
    # Load raw data
    df = pd.read_csv(config.movielens_path)
    
    # Preprocess
    df['user_id'] = df['userId'].astype('category').cat.codes
    df['item_id'] = df['movieId'].astype('category').cat.codes
    df['timestamp'] = df['timestamp'].astype(int)
    df['rating'] = (df['rating'] > 3.5).astype(int)  # Convert ratings to binary feedback
    
    return create_train_val_test_split(df, config)

def load_amazon_dataset(config):
    """
    Load and preprocess the Amazon Electronics dataset.
    """
    # Load raw data
    df = pd.read_json(config.amazon_path, lines=True)
    
    # Preprocess
    df['user_id'] = df['reviewerID'].astype('category').cat.codes
    df['item_id'] = df['asin'].astype('category').cat.codes
    df['timestamp'] = pd.to_datetime(df['unixReviewTime'], unit='s').astype(int) / 10**9
    df['rating'] = (df['overall'] > 3).astype(int)  # Convert ratings to binary feedback
    
    return create_train_val_test_split(df, config)

def create_train_val_test_split(df, config):
    """
    Create train, validation, and test splits.
    """
    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Split into train+val and test
    train_val, test = train_test_split(df, test_size=config.test_size, shuffle=False)

    # Split train+val into train and val
    train, val = train_test_split(train_val, test_size=config.val_size, shuffle=False)

    # Create DataLoader objects
    train_dataset = PPTEBDEDataset(train)
    val_dataset = PPTEBDEDataset(val)
    test_dataset = PPTEBDEDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
